import Foundation
import Testing
import NVMAI
import NVMAIMemory
@testable import NVMAIServerCore

/// The engine writing memory on its own, and the loop answering when its
/// rounds are gone. Both exist because a real model was measured not doing
/// the thing the design assumed it would.
@Suite struct MemoryConsolidationTests {
    /// unchecked-invariant: every access to `script` and `seen` is under `lock`.
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
            return completion
        }
        var requests: [ValidatedChatRequest] { lock.withLock { seen } }
    }

    private func completion(_ content: String, calls: [ParsedToolCall] = [],
                            finish: String = "stop") -> ServerCompletion {
        ServerCompletion(content: content, toolCalls: calls, finishReason: finish,
                         usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    private func call(_ name: String, _ arguments: [String: String]) -> ParsedToolCall {
        ParsedToolCall(id: "call-\(name)-\(UUID().uuidString.prefix(4))", name: name,
                       arguments: .object(arguments.mapValues { .string($0) }),
                       argumentsJSON: "{}")
    }

    private func request(_ text: String) -> ValidatedChatRequest {
        ValidatedChatRequest(messages: [GFTokenizer.Message(role: .user, content: text)],
                             tools: [], stream: false, includeUsage: false,
                             generationConfig: GenerationConfig(maxNewTokens: 32),
                             maximumCompletionTokens: 32)
    }

    private func configuration(rounds: Int = 2,
                               tools: MemoryToolSurface = .full,
                               consolidation: Bool,
                               idleSeconds: Double = 0.05) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.maximumToolRounds = rounds
        configuration.toolSurface = tools
        configuration.sessionConsolidation = consolidation
        configuration.consolidationIdleSeconds = idleSeconds
        // These sessions are a line each; the trivial-session guard has its
        // own test and would otherwise skip every one of them.
        configuration.consolidationMinimumCharacters = 0
        return configuration
    }

    // MARK: - Round exhaustion

    /// Measured: a model that wanted memory on every round got its preamble
    /// returned as the answer -- 31 tokens where ten chapters should have
    /// been. Now the last calls are answered and it gets one more turn.
    @Test func exhaustedRoundsEndInAnAnswerNotAPreamble() async throws {
        let inner = ScriptedBackend([
            completion("I need to check memory first.", calls: [call("memory_get", ["key": "a"])]),
            completion("Still checking.", calls: [call("memory_get", ["key": "b"])]),
            // Rounds are gone and the model still asks. This is where the
            // old loop returned the preamble above as the answer.
            completion("One more look.", calls: [call("memory_get", ["key": "c"])]),
            // The final, tool-free generation.
            completion("Chapter 11\nThe tide came in."),
        ])
        let configuration = configuration(rounds: 2, consolidation: false)
        let service = MemoryService(configuration: configuration,
                                    durableStore: InMemoryStore())
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        let result = try await backend.generate(request("write chapter 11"), onEvent: { _ in })
        #expect(result.content.contains("The tide came in."))
        #expect(result.finishReason == "stop")
        #expect(result.toolCalls.isEmpty)

        // Four generations: two rounds, the one that overran, and the answer.
        // The last request carries the exhaustion notice and still carries
        // the tools, so the prompt prefix did not move.
        let requests = inner.requests
        #expect(requests.count == 4)
        let last = try #require(requests.last)
        #expect(last.messages.last?.role == .user)
        #expect(last.messages.last?.content?.contains("used up") == true)
        #expect(last.tools.contains { $0.name == "memory_get" })
    }

    @Test func aToolCallOnTheFinalTurnIsDroppedRatherThanLooped() async throws {
        let inner = ScriptedBackend([
            completion("", calls: [call("memory_get", ["key": "a"])]),
            completion("", calls: [call("memory_get", ["key": "b"])]),
            completion("", calls: [call("memory_get", ["key": "c"])]),
            // The final turn answers and, wrongly, calls a tool as well.
            completion("Here is the answer anyway.", calls: [call("memory_get", ["key": "d"])]),
        ])
        let configuration = configuration(rounds: 2, consolidation: false)
        let service = MemoryService(configuration: configuration,
                                    durableStore: InMemoryStore())
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)
        let result = try await backend.generate(request("go"), onEvent: { _ in })
        #expect(result.content == "Here is the answer anyway.")
        #expect(result.toolCalls.isEmpty)
        #expect(inner.requests.count == 4)
    }

    // MARK: - Consolidation

    private static let extraction = """
    ```json
    [
      {"key": "characters/marcus", "value": "Marcus has grey eyes", "importance": 0.9},
      {"key": "state/inn", "value": "The inn burned down in chapter 34", "importance": 0.8},
      {"key": "Bad Key!", "value": "skipped", "importance": 0.1}
    ]
    ```
    """

    /// After a turn, the idle timer fires and the engine writes facts the
    /// model never chose to write.
    @Test func theIdleTimerDistilsTheSessionIntoMemory() async throws {
        let inner = ScriptedBackend([
            completion("Chapter 34: the inn burned."),   // the turn
            completion(Self.extraction),                 // the consolidation
        ])
        let configuration = configuration(tools: .off, consolidation: true)
        let store = InMemoryStore()
        let service = MemoryService(configuration: configuration, durableStore: store,
                                    journal: InMemoryJournal())
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)
        _ = try await backend.generate(request("write chapter 34"), onEvent: { _ in })

        let scope = try MemoryScope(namespace: "nvmai", user: "local", workspace: "repo-a")
        var written: [MemoryRecord] = []
        for _ in 0..<100 {
            try await Task.sleep(for: .milliseconds(25))
            written = await store.allRecords(in: scope)
            if !written.isEmpty { break }
        }
        #expect(written.count == 2, "two valid facts, the malformed key skipped")
        #expect(written.contains { $0.key.rawValue == "state/inn" && $0.value.contains("burned") })

        // The extraction request is its own conversation: no tools, a system
        // message that names the job, and the session's transcript.
        let extraction = try #require(inner.requests.last)
        #expect(extraction.tools.isEmpty)
        #expect(extraction.messages.first?.role == .system)
        #expect(extraction.messages.last?.content?.contains("the inn burned") == true)
        await backend.shutDown()
    }

    /// A new session before the timer fires is a rollover. The previous
    /// session is consolidated after the new session's request returns --
    /// never before it, because a person is waiting on that request.
    @Test func aRolloverConsolidatesThePreviousSessionAfterTheTurn() async throws {
        let inner = ScriptedBackend([
            completion("Chapter 1."),                    // session A, turn
            completion("Chapter 11."),                   // session B, turn (rollover)
            completion(Self.extraction),                 // consolidation of A
        ])
        // A long idle so only the rollover can trigger it.
        let configuration = configuration(tools: .off, consolidation: true, idleSeconds: 60)
        let store = InMemoryStore()
        let service = MemoryService(configuration: configuration, durableStore: store,
                                    journal: InMemoryJournal())
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        _ = try await backend.generate(request("write chapter 1"), onEvent: { _ in })
        #expect(inner.requests.count == 1, "nothing consolidates while the session is live")
        let second = try await backend.generate(request("write chapter 11"), onEvent: { _ in })
        #expect(second.content == "Chapter 11.", "the rollover request got its own answer")

        let scope = try MemoryScope(namespace: "nvmai", user: "local", workspace: "repo-a")
        var written: [MemoryRecord] = []
        for _ in 0..<100 {
            try await Task.sleep(for: .milliseconds(25))
            written = await store.allRecords(in: scope)
            if !written.isEmpty { break }
        }
        #expect(written.count == 2)
        // The extraction came after session B's answer, and it was A's
        // transcript that was distilled.
        #expect(inner.requests.count == 3)
        #expect(inner.requests.last?.messages.last?.content?.contains("chapter 1") == true)
        await backend.shutDown()
    }

    @Test func consolidationOffMeansNoExtraGeneration() async throws {
        let inner = ScriptedBackend([completion("done")])
        let configuration = configuration(tools: .off, consolidation: false)
        let service = MemoryService(configuration: configuration, durableStore: InMemoryStore(),
                                    journal: InMemoryJournal())
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)
        _ = try await backend.generate(request("hi"), onEvent: { _ in })
        try await Task.sleep(for: .milliseconds(200))
        #expect(inner.requests.count == 1)
    }

    // MARK: - The parser

    @Test func extractionOutputIsParsedLeniently() {
        let fenced = ServerMemory.consolidationRecords(from: Self.extraction)
        #expect(fenced.map(\.key.rawValue) == ["characters/marcus", "state/inn"])
        #expect(fenced.first?.importance == 0.9)

        let bare = ServerMemory.consolidationRecords(
            from: "Here you go: [{\"key\": \"decisions/x\", \"value\": 42}] hope that helps")
        #expect(bare.first?.key.rawValue == "decisions/x")
        #expect(bare.first?.value == "42")

        #expect(ServerMemory.consolidationRecords(from: "[]").isEmpty)
        #expect(ServerMemory.consolidationRecords(from: "no json here").isEmpty)
        #expect(ServerMemory.consolidationRecords(from: "[{\"value\": \"no key\"}]").isEmpty)
    }

    @Test func minimalSurfaceIncludesList() {
        #expect(MemoryToolSurface.minimal.toolNames == ["memory_set", "memory_get", "memory_list"])
    }

    @Test func consolidationIsOnByDefaultWithMemory() {
        let on = MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY": "1"])
        #expect(on.sessionConsolidation)
        #expect(on.consolidationIdleSeconds == 120)
        let off = MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY": "1",
                                                        "NVMAI_MEMORY_CONSOLIDATION": "0"])
        #expect(!off.sessionConsolidation)
        let quick = MemoryConfiguration.fromEnvironment(
            ["NVMAI_MEMORY": "1", "NVMAI_MEMORY_CONSOLIDATION_IDLE_SECONDS": "5"])
        #expect(quick.consolidationIdleSeconds == 5)
    }
}

/// The generation gate: a consolidation may never overlap a person's turn.
@Suite struct MemoryGenerationGateTests {
    /// A backend that fails the test if it is ever entered twice at once.
    /// unchecked-invariant: `active`, `peak` and `calls` are only touched
    /// under `lock`.
    private final class OverlapDetector: ServerInferenceBackend, @unchecked Sendable {
        private let lock = NSLock()
        private var active = 0
        private(set) var peak = 0
        private(set) var calls = 0
        private let reply: @Sendable (Int) -> String
        init(reply: @escaping @Sendable (Int) -> String) { self.reply = reply }
        func generate(_ request: ValidatedChatRequest,
                      onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws
            -> ServerCompletion {
            let index: Int = lock.withLock {
                active += 1; peak = max(peak, active); calls += 1; return calls
            }
            try await Task.sleep(for: .milliseconds(60))
            lock.withLock { active -= 1 }
            return ServerCompletion(content: reply(index), toolCalls: [], finishReason: "stop",
                                    usage: OpenAIUsage(promptTokens: 1, completionTokens: 1,
                                                       totalTokens: 2))
        }
        var maximumConcurrency: Int { lock.withLock { peak } }
        var callCount: Int { lock.withLock { calls } }
    }

    private func request(_ text: String) -> ValidatedChatRequest {
        ValidatedChatRequest(messages: [GFTokenizer.Message(role: .user, content: text)],
                             tools: [], stream: false, includeUsage: false,
                             generationConfig: GenerationConfig(maxNewTokens: 32),
                             maximumCompletionTokens: 32)
    }

    @Test func consolidationNeverOverlapsATurn() async throws {
        let extraction = "```json\n[{\"key\": \"a/b\", \"value\": \"c\", \"importance\": 0.5}]\n```"
        let long = String(repeating: "a substantial reply. ", count: 40)
        // Turns get long replies (so the session is worth distilling);
        // consolidations get the extraction.
        let inner = OverlapDetector { index in index % 2 == 1 ? long : extraction }
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .off
        configuration.sessionConsolidation = true
        configuration.consolidationIdleSeconds = 0.01
        let service = MemoryService(configuration: configuration, durableStore: InMemoryStore(),
                                    journal: InMemoryJournal())
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)

        // Fire turns fast enough that each idle timer lands while the next
        // turn is running.
        for index in 0..<6 {
            _ = try await backend.generate(request("turn \(index) " + long), onEvent: { _ in })
            try await Task.sleep(for: .milliseconds(15))
        }
        try await Task.sleep(for: .milliseconds(400))
        #expect(inner.maximumConcurrency == 1, "a consolidation overlapped a turn")
        #expect(inner.callCount > 6, "no consolidation ever ran")
        await backend.shutDown()
    }

    @Test func trivialSessionsAreNotDistilled() async throws {
        let inner = OverlapDetector { _ in "OK" }
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .off
        configuration.sessionConsolidation = true
        configuration.consolidationIdleSeconds = 0.01
        let service = MemoryService(configuration: configuration, durableStore: InMemoryStore(),
                                    journal: InMemoryJournal())
        let backend = MemoryBackend(wrapping: inner, service: service,
                                    configuration: configuration)
        _ = try await backend.generate(request("Say OK."), onEvent: { _ in })
        try await Task.sleep(for: .milliseconds(300))
        // One generation: the turn. A "say OK" session buys no second one.
        #expect(inner.callCount == 1)
        await backend.shutDown()
    }
}

/// What a hundred-chapter run taught the extraction.
@Suite struct MemoryConsolidationExtractionTests {
    @Test func aTruncatedArrayStillYieldsTheCompleteObjects() {
        // The output cap landed inside the seventeenth object.
        let cut = """
        ```json
        [
          {"key": "characters/marcus/eyes", "value": "grey", "importance": 0.9},
          {"key": "state/inn", "value": "burned down in chapter 34", "importance": 0.8},
          {"key": "state/tomas", "value": "found alive in the ligh
        """
        let records = ServerMemory.consolidationRecords(from: cut)
        #expect(records.map(\.key.rawValue) == ["characters/marcus/eyes", "state/inn"])
    }

    @Test func placeholdersAreNeverWrittenOverAFact() {
        let output = """
        [
          {"key": "characters/halvorsen/eyes", "value": "not specified", "importance": 0.5},
          {"key": "state/ferry_day", "value": "N/A", "importance": 0.5},
          {"key": "state/ferry_running", "value": false, "importance": 0.7},
          {"key": "state/anyone_left", "value": null, "importance": 0.7},
          {"key": "rules/weather", "value": "it never rains", "importance": 0.9}
        ]
        """
        let records = ServerMemory.consolidationRecords(from: output)
        #expect(records.map(\.key.rawValue) == ["state/ferry_running", "rules/weather"])
        #expect(records.first?.value == "false")
    }

    @Test func thePromptShowsValuesAndAsksOnlyForChanges() throws {
        let existing = [
            MemoryRecord(key: try MemoryKey(validating: "characters/marcus/eyes"), value: "grey"),
            MemoryRecord(key: try MemoryKey(validating: "state/inn"), value: "standing"),
        ]
        let turn = JournalTurn(session: "s", workspace: "w", index: 0,
                               prompt: "write chapter 34", reply: "The inn burned.")
        let request = ServerMemory.consolidationRequest(turns: [turn], existing: existing,
                                                        workspace: "w")
        let user = request.messages.last?.content ?? ""
        let system = request.messages.first?.content ?? ""
        #expect(user.contains("state/inn = standing"))
        #expect(user.contains("characters/marcus/eyes = grey"))
        #expect(system.contains("ONLY facts this session added or changed"))
        #expect(system.contains("omit the key instead"))
        #expect(request.maximumCompletionTokens >= 2000)
        #expect(request.tools.isEmpty)
    }
}
