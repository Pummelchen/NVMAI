import Foundation
import NVMAI
import NVMAIMemory

/// Adds persistent memory to any inference backend.
///
/// A decorator rather than a change to generation: it rewrites the request on
/// the way in (the memory instructions and the memory tools), and on the way
/// out it services the memory tool calls the model makes and asks the inner
/// backend to continue. The engine's request lifecycle, prompt cache and
/// tool parsing are untouched, and with memory disabled this type is not
/// constructed at all.
///
/// Why the engine executes these tools when it executes no others: the
/// server's own tools are the client's, and no client knows about NVMAI
/// memory. A memory tool the client would have to run is a memory tool
/// nothing runs.
public actor MemoryBackend: ServerInferenceBackend {
    private let inner: any ServerInferenceBackend
    private let service: MemoryService
    private let configuration: MemoryConfiguration
    /// Session contexts by conversation, so a multi-turn conversation keeps
    /// one session and bootstraps once.
    private var contexts: [String: MemorySessionContext] = [:]

    public init(wrapping inner: any ServerInferenceBackend,
                service: MemoryService,
                configuration: MemoryConfiguration) {
        self.inner = inner
        self.service = service
        self.configuration = configuration
    }

    public nonisolated var maximumContext: Int { inner.maximumContext }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { inner.samplingDefaults }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        guard let context = await sessionContext(for: request) else {
            return try await inner.generate(request, onEvent: onEvent)
        }
        let instructions = await service.instructions(for: context)
        let memoryTools = ServerMemory.functionDefinitions(await service.toolDefinitions())

        var current = request.replacingMessages(
            ConcisePrompt.appendingSystemPrompt(instructions, to: request.messages),
            tools: ServerMemory.merging(tools: request.tools, memory: memoryTools))

        // Memory tool calls are ours to answer, so the client never sees
        // them; anything else, including the client's own tools, passes
        // through untouched.
        let filteredEvents: @Sendable (ServerInferenceEvent) -> Void = { event in
            if case .toolCall(let call) = event, MemoryTools.isMemoryTool(call.name) { return }
            onEvent(event)
        }

        var transcript = ""
        var rounds = 0
        while true {
            let completion = try await inner.generate(current, onEvent: filteredEvents)
            let memoryCalls = completion.toolCalls.filter { MemoryTools.isMemoryTool($0.name) }
            let otherCalls = completion.toolCalls.filter { !MemoryTools.isMemoryTool($0.name) }
            transcript += completion.content

            // Stop when the model is done with memory. A turn that also calls
            // a client tool ends here as well: the client has to run that one,
            // and continuing would strand its result.
            guard !memoryCalls.isEmpty, otherCalls.isEmpty, rounds < configuration.maximumToolRounds
            else {
                return ServerCompletion(content: transcript,
                                        toolCalls: otherCalls,
                                        finishReason: memoryCalls.isEmpty
                                            ? completion.finishReason
                                            : roundLimitReason(completion, memoryCalls, rounds),
                                        usage: completion.usage)
            }

            rounds += 1
            var messages = current.messages
            messages.append(ServerMemory.assistantMessage(content: completion.content,
                                                          calls: memoryCalls))
            for call in memoryCalls {
                let result = await service.execute(
                    name: call.name,
                    arguments: ServerMemory.arguments(from: call.arguments),
                    in: context)
                messages.append(ServerMemory.toolResultMessage(call: call, result: result))
                ServerLog.memory("tool=\(call.name) "
                                 + (result.isFailure ? "failed" : "ok")
                                 + " round=\(rounds) session=\(context.session.id)")
            }
            current = current.replacingMessages(messages, tools: current.tools)
        }
    }

    /// Runs the session-end hook, if consolidation is on. The engine calls
    /// this when a conversation is finished with; nothing calls it
    /// automatically, because the API has no end-of-conversation signal.
    public func endSession(conversation id: String) async {
        guard let context = contexts.removeValue(forKey: id) else { return }
        await service.endSession(context)
    }

    /// Resolves, and caches, the memory session for this conversation.
    private func sessionContext(for request: ValidatedChatRequest) async
        -> MemorySessionContext? {
        let workspace = request.workspace ?? configuration.workspace
        let id = ServerMemory.sessionIdentifier(messages: request.messages, workspace: workspace)
        if let existing = contexts[id] { return existing }
        guard let context = await service.beginSession(
            id: id,
            workspaceOverride: request.workspace,
            modelID: nil) else { return nil }
        contexts[id] = context
        ServerLog.memory("session=\(context.session.id) scope=\(context.scope.workspace) "
                         + "bootstrap=\(context.bootstrap.records.count) "
                         + "durable=\(context.isDurable)")
        return context
    }

    /// When the round limit stops a conversation mid-memory, say so in the
    /// finish reason rather than presenting a truncated answer as complete.
    private func roundLimitReason(_ completion: ServerCompletion,
                                  _ calls: [ParsedToolCall],
                                  _ rounds: Int) -> String {
        rounds >= configuration.maximumToolRounds ? "length" : completion.finishReason
    }
}

/// Builds the memory decorator, or returns the backend unchanged.
///
/// The command target calls this so it never has to know how the service is
/// assembled or how memory logs; with memory off it is a pass-through and no
/// memory type is constructed.
public enum ServerMemoryFactory {
    public static func wrap(_ backend: any ServerInferenceBackend,
                            configuration: MemoryConfiguration = .fromEnvironment())
        -> any ServerInferenceBackend {
        guard configuration.isEnabled else { return backend }
        let service = MemoryService(configuration: configuration) { event in
            ServerLog.memory(event.message)
        }
        ServerLog.memory(configuration.summary)
        return MemoryBackend(wrapping: backend,
                             service: service,
                             configuration: configuration)
    }
}
