import Foundation
import NVMAI
import NVMAIMemory
import Testing
@testable import NVMAIServerCore

/// The decorator is where memory meets the request lifecycle: what the model
/// is shown, what the client is shown, and what the engine runs on the
/// model's behalf.
@Suite struct MemoryBackendTests {
    /// A backend that replays scripted completions and records what it was
    /// asked to generate.
    ///
    /// unchecked-invariant: every member is guarded by `lock`.
    private final class ScriptedBackend: ServerInferenceBackend, @unchecked Sendable {
        private var script: [ServerCompletion]
        private var seen: [ValidatedChatRequest] = []
        private let lock = NSLock()

        init(_ script: [ServerCompletion]) { self.script = script }

        func generate(_ request: ValidatedChatRequest,
                      onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws
            -> ServerCompletion {
            lock.withLock { seen.append(request) }
            let completion = lock.withLock { script.isEmpty ? nil : script.removeFirst() }
            guard let completion else {
                return ServerCompletion(content: "", toolCalls: [], finishReason: "stop",
                                        usage: OpenAIUsage(promptTokens: 0, completionTokens: 0,
                                                           totalTokens: 0))
            }
            if !completion.content.isEmpty { onEvent(.content(completion.content)) }
            for call in completion.toolCalls { onEvent(.toolCall(call)) }
            return completion
        }

        var requests: [ValidatedChatRequest] { lock.withLock { seen } }
        var callCount: Int { lock.withLock { seen.count } }
    }

    /// Collects streamed events from the `@Sendable` callback.
    ///
    /// unchecked-invariant: every access is under `lock`.
    private final class EventSink: @unchecked Sendable {
        private var events: [ServerInferenceEvent] = []
        private let lock = NSLock()
        func append(_ event: ServerInferenceEvent) { lock.withLock { events.append(event) } }
        var all: [ServerInferenceEvent] { lock.withLock { events } }
        var toolCalls: [ParsedToolCall] {
            all.compactMap { if case .toolCall(let call) = $0 { return call } else { return nil } }
        }
    }

    private func completion(_ content: String,
                            calls: [ParsedToolCall] = [],
                            finish: String = "stop") -> ServerCompletion {
        ServerCompletion(content: content, toolCalls: calls, finishReason: finish,
                         usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    private func memoryCall(_ name: String, _ arguments: [String: JSONValue]) -> ParsedToolCall {
        ParsedToolCall(id: "call-\(name)", name: name, arguments: .object(arguments),
                       argumentsJSON: "{}")
    }

    private func request(_ text: String = "hello",
                         stream: Bool = false) -> ValidatedChatRequest {
        ValidatedChatRequest(messages: [GFTokenizer.Message(role: .user, content: text)],
                             tools: [],
                             stream: stream,
                             includeUsage: false,
                             generationConfig: GenerationConfig(maxNewTokens: 32),
                             maximumCompletionTokens: 32)
    }

    private func service(store: any MemoryStore = InMemoryStore(),
                        workspace: String = "repo-a",
                        rounds: Int = 4) -> (MemoryService, MemoryConfiguration) {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = workspace
        configuration.user = "local"
        configuration.maximumToolRounds = rounds
        return (MemoryService(configuration: configuration, durableStore: store), configuration)
    }

    @Test func installsInstructionsAndToolsWithoutTouchingTheUserMessage() async throws {
        let inner = ScriptedBackend([completion("hi")])
        let (service, configuration) = service()
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        let seen = try #require(inner.requests.first)
        // The fragment goes in as a system message; the user's own message is
        // untouched.
        let system = seen.messages.first { $0.role == .system }?.content ?? ""
        #expect(system.contains("Persistent memory"))
        #expect(system.contains("memory_search"))
        #expect(seen.messages.last?.role == .user)
        #expect(seen.messages.last?.content == "hello")
        #expect(Set(seen.tools.map(\.name)) == MemoryTools.names)
    }

    @Test func bootstrapNeverCarriesTheWholeStore() async throws {
        let store = InMemoryStore(limits: MemoryLimits(bootstrapRecords: 3, bootstrapBytes: 400))
        let scope = try MemoryScope(namespace: "nvmai", user: "local", workspace: "repo-a")
        for index in 0..<50 {
            try await store.set(MemoryRecord(key: try MemoryKey(validating: "facts/f\(index)"),
                                             value: String(repeating: "v", count: 100),
                                             importance: Double(index) / 50), in: scope)
        }
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.limits = MemoryLimits(bootstrapRecords: 3, bootstrapBytes: 400)
        let inner = ScriptedBackend([completion("hi")])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(configuration: configuration, durableStore: store),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        let system = try #require(inner.requests.first?.messages
            .first { $0.role == .system }?.content)
        // Three keys named, the rest counted, and no room for 50 values.
        #expect(system.contains("and 47 more"))
        #expect(system.utf8.count < 2_000)
    }

    @Test func executesMemoryCallsAndContinuesWithoutTellingTheClient() async throws {
        let store = InMemoryStore()
        let inner = ScriptedBackend([
            completion("Let me check.", calls: [memoryCall("memory_set", [
                "key": .string("decisions/sync"),
                "value": .string("Keep FooManager; it prevents a background sync race."),
            ])]),
            completion(" Stored."),
        ])
        let (service, configuration) = service(store: store)
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        let events = EventSink()
        let completion = try await backend.generate(request()) { events.append($0) }

        // The engine ran the tool and the model continued; the client saw
        // text only, never a tool call it cannot execute.
        #expect(inner.callCount == 2)
        #expect(completion.content == "Let me check. Stored.")
        #expect(completion.toolCalls.isEmpty)
        #expect(events.toolCalls.isEmpty)

        let scope = try MemoryScope(namespace: "nvmai", user: "local", workspace: "repo-a")
        let stored = try await store.get(try MemoryKey(validating: "decisions/sync"), in: scope)
        #expect(stored?.value.contains("FooManager") == true)
        #expect(stored?.sourceSession?.isEmpty == false)

        // The continuation turn carries the assistant call and the tool result.
        let second = try #require(inner.requests.last)
        #expect(second.messages.contains { $0.role == .tool && $0.name == "memory_set" })
        #expect(second.messages.contains { $0.role == .assistant && !$0.toolCalls.isEmpty })
    }

    @Test func clientToolCallsPassThroughUntouched() async throws {
        let clientCall = ParsedToolCall(id: "c1", name: "read_file",
                                        arguments: .object(["path": .string("a.swift")]),
                                        argumentsJSON: "{}")
        let inner = ScriptedBackend([completion("Reading.", calls: [clientCall])])
        let (service, configuration) = service()
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        let events = EventSink()
        let completion = try await backend.generate(request()) { events.append($0) }

        // The client's own tools are still the client's to run.
        #expect(inner.callCount == 1)
        #expect(completion.toolCalls.map(\.name) == ["read_file"])
        #expect(events.toolCalls.map(\.name) == ["read_file"])
    }

    @Test func stopsAtTheRoundLimit() async throws {
        // A model that only ever calls memory must not loop forever.
        let call = memoryCall("memory_list", [:])
        let inner = ScriptedBackend(Array(repeating: completion("...", calls: [call]), count: 10))
        let (service, configuration) = service(rounds: 2)
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        let completion = try await backend.generate(request(), onEvent: { _ in })
        #expect(inner.callCount == 3)
        #expect(completion.finishReason == "length")
    }

    @Test func failedMemoryWritesAreReportedToTheModel() async throws {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.limits.maximumValueBytes = 16
        let inner = ScriptedBackend([
            completion("", calls: [memoryCall("memory_set", [
                "key": .string("facts/big"),
                "value": .string(String(repeating: "x", count: 64)),
            ])]),
            completion("I could not save that."),
        ])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(configuration: configuration,
                                   durableStore: InMemoryStore(limits: configuration.limits)),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        // The model has to learn the write failed, or it will report a fact
        // as saved that is not.
        let toolMessage = try #require(inner.requests.last?.messages
            .first { $0.role == .tool }?.content)
        #expect(toolMessage.contains("\"ok\":false"))
    }

    @Test func requestsInDifferentWorkspacesGetDifferentScopes() async throws {
        let store = InMemoryStore()
        let inner = ScriptedBackend([
            completion("", calls: [memoryCall("memory_set", ["key": .string("facts/a"),
                                                             "value": .string("from-a")])]),
            completion("ok"),
            completion("", calls: [memoryCall("memory_set", ["key": .string("facts/a"),
                                                             "value": .string("from-b")])]),
            completion("ok"),
        ])
        let (service, configuration) = service(store: store)
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        _ = try await backend.generate(request("one").withWorkspace("repo-a"), onEvent: { _ in })
        _ = try await backend.generate(request("two").withWorkspace("repo-b"), onEvent: { _ in })

        let a = try MemoryScope(namespace: "nvmai", user: "local", workspace: "repo-a")
        let b = try MemoryScope(namespace: "nvmai", user: "local", workspace: "repo-b")
        #expect(try await store.get(try MemoryKey(validating: "facts/a"), in: a)?.value == "from-a")
        #expect(try await store.get(try MemoryKey(validating: "facts/a"), in: b)?.value == "from-b")
    }

    @Test func disabledMemoryLeavesTheRequestAlone() async throws {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = false
        let inner = ScriptedBackend([completion("hi")])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(configuration: configuration),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        let seen = try #require(inner.requests.first)
        #expect(!seen.messages.contains { $0.role == .system })
        #expect(seen.tools.isEmpty)
    }

    @Test func oneConversationBootstrapsOnceAcrossTurns() async throws {
        let inner = ScriptedBackend([completion("a"), completion("b")])
        let (service, configuration) = service()
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        let first = request("same opening question")
        _ = try await backend.generate(first, onEvent: { _ in })
        // A second turn of the same conversation: the client resends the
        // history, so the session id has to stay stable.
        var messages = first.messages
        messages.append(GFTokenizer.Message(role: .assistant, content: "a"))
        messages.append(GFTokenizer.Message(role: .user, content: "follow up"))
        _ = try await backend.generate(first.replacingMessages(messages, tools: []),
                                       onEvent: { _ in })

        let ids = inner.requests.compactMap { seen in
            seen.messages.first { $0.role == .system }?.content
        }
        #expect(ids.count == 2)
        #expect(inner.callCount == 2)
    }
}
