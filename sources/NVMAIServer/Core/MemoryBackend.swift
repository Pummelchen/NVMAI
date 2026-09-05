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
    /// The exact instruction text installed for a conversation, kept for the
    /// life of that conversation.
    ///
    /// This is the constraint the whole design turns on. The fragment sits at
    /// the head of the prompt, so if it changed between turns the prefix
    /// would change and every cached KV block after it would be invalid.
    /// NVMAI prefills at roughly twice its decode rate rather than the
    /// hundredfold of a GPU server, so a needless cache miss costs minutes,
    /// not milliseconds. The bootstrap is therefore computed once per
    /// conversation and frozen, even though memory keeps changing underneath
    /// it: a stale bootstrap is cheap, and the model can always call a tool
    /// or read the journal for what is current.
    private var installedInstructions: [String: String] = [:]
    /// Turn counter per conversation, for the journal.
    private var turnIndex: [String: Int] = [:]
    /// Read once. The workspace guard needs it per request, and reading the
    /// environment per request is the pattern that once cost 40% of a token.
    private let homeDirectory = FileManager.default.homeDirectoryForCurrentUser.path
    /// Declared directories already refused, so each is logged once.
    private var refusedDirectories: Set<String> = []
    /// Sessions with turns not yet distilled into memory, by scope. One per
    /// scope: a new session in a scope replaces the pending one, and the
    /// replaced one is consolidated on the rollover it just caused.
    private var unconsolidated: [MemoryScope: MemorySessionContext] = [:]
    /// The idle timer per scope. Reset on every turn; fires consolidation.
    private var idleTimers: [MemoryScope: Task<Void, Never>] = [:]
    /// A session that rolled over before its idle timer fired. Consolidated
    /// as soon as the current request has returned, never before.
    private var pendingAfterTurn: [MemoryScope: MemorySessionContext] = [:]

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
        let conversation = conversationKey(for: request)
        // Frozen on the first turn and reused verbatim thereafter, so the
        // prompt prefix is stable for the life of the conversation.
        let instructions: String
        if let existing = installedInstructions[conversation] {
            instructions = existing
        } else {
            instructions = await service.instructions(for: context)
            installedInstructions[conversation] = instructions
        }
        let memoryTools = ServerMemory.functionDefinitions(await service.toolDefinitions())

        var current = request.replacingMessages(
            ConcisePrompt.appendingSystemPrompt(instructions, to: request.messages),
            tools: ServerMemory.merging(tools: request.tools, memory: memoryTools))
        let startedAt = Date()

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
            guard !memoryCalls.isEmpty, otherCalls.isEmpty else {
                let finished = ServerCompletion(content: transcript,
                                                toolCalls: otherCalls,
                                                finishReason: completion.finishReason,
                                                usage: completion.usage)
                await journal(request: request, completion: finished, context: context,
                              conversation: conversation, startedAt: startedAt)
                return finished
            }

            // Rounds exhausted and the model still wants memory. Returning
            // here would hand back whatever preamble preceded the last call --
            // measured, that was a 31-token "I need to check the existing
            // memories" where ten chapters should have been. Instead the last
            // calls are answered, the model is told the rounds are gone, and
            // it gets one more generation to answer with what it has. The
            // tools stay in the request so the prompt prefix does not move;
            // any tool call it makes anyway is dropped.
            if rounds >= configuration.maximumToolRounds {
                var messages = current.messages
                messages.append(ServerMemory.assistantMessage(content: completion.content,
                                                              calls: memoryCalls))
                for call in memoryCalls {
                    let result = await service.execute(
                        name: call.name,
                        arguments: ServerMemory.arguments(from: call.arguments),
                        in: context)
                    messages.append(ServerMemory.toolResultMessage(call: call, result: result))
                }
                messages.append(GFTokenizer.Message(
                    role: .user,
                    content: "Your memory tool rounds for this turn are used up. Answer the "
                        + "original request now, in full, without calling any tools."))
                current = current.replacingMessages(messages, tools: current.tools)
                let last = try await inner.generate(current, onEvent: filteredEvents)
                transcript += last.content
                let finished = ServerCompletion(
                    content: transcript,
                    toolCalls: last.toolCalls.filter { !MemoryTools.isMemoryTool($0.name) },
                    finishReason: transcript.isEmpty ? "length" : last.finishReason,
                    usage: last.usage)
                ServerLog.memory("tool rounds exhausted; answered without tools "
                                 + "session=\(context.session.id)")
                await journal(request: request, completion: finished, context: context,
                              conversation: conversation, startedAt: startedAt)
                return finished
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

    /// Writes the turn to the journal after the completion is settled.
    ///
    /// This runs after generation and before the completion is returned, so
    /// it is on the request path -- but only for a `write(2)` into the page
    /// cache, which is microseconds. The durability barrier is deliberately
    /// not here: the journal takes it a couple of seconds after the last
    /// append, once the drive is idle, so nothing this does can hold the
    /// answer or contend with the expert streamer. The journal swallows its
    /// own failures. Only the user's prompt and the assistant's reply text
    /// go in; tool definitions, tool calls and tool results never reach it,
    /// which is what keeps a turn at a few kilobytes.
    private func journal(request: ValidatedChatRequest,
                         completion: ServerCompletion,
                         context: MemorySessionContext,
                         conversation: String,
                         startedAt: Date) async {
        let index = (turnIndex[conversation] ?? 0)
        turnIndex[conversation] = index + 1
        let prompt = request.messages.last { $0.role == .user }?.content ?? ""
        await service.recordTurn(
            session: context,
            index: index,
            prompt: prompt,
            reply: completion.content,
            model: nil,
            promptTokens: completion.usage.promptTokens,
            completionTokens: completion.usage.completionTokens,
            latencyMilliseconds: Int(Date().timeIntervalSince(startedAt) * 1_000),
            stopReason: completion.finishReason)
        scheduleConsolidation(after: context)
    }

    // MARK: - Consolidation

    /// Arms the idle timer for a session that just gained a turn, and fires
    /// the consolidation of a session that rolled over during this request.
    ///
    /// The turn is recorded and the reply has been returned by the time this
    /// runs, so the person is reading. That is the pause a consolidation is
    /// allowed to use.
    private func scheduleConsolidation(after context: MemorySessionContext) {
        guard configuration.sessionConsolidation else { return }
        let scope = context.scope
        unconsolidated[scope] = context
        idleTimers[scope]?.cancel()
        let delay = configuration.consolidationIdleSeconds
        idleTimers[scope] = Task { [weak self] in
            if delay > 0 { try? await Task.sleep(for: .seconds(delay)) }
            guard !Task.isCancelled, let self else { return }
            await self.consolidateIfPending(scope: scope, expecting: context.session.id)
        }
        if let previous = pendingAfterTurn.removeValue(forKey: scope) {
            Task { [weak self] in await self?.consolidate(previous) }
        }
    }

    private func consolidateIfPending(scope: MemoryScope, expecting sessionID: String) async {
        guard let pending = unconsolidated[scope], pending.session.id == sessionID else { return }
        await consolidate(pending)
    }

    /// Distils a finished session into memory.
    ///
    /// This is the engine writing, not the model choosing to. Measured on a
    /// hundred-chapter novel, a model given the bible in its prompt made zero
    /// writes in that session, then found memory empty in the next and
    /// stored a bible it had invented; a harness that simply forced a
    /// summary at each boundary carried twice as much. The forcing is what
    /// works. Writing the result as addressed facts rather than a note is
    /// what lets a later change supersede an earlier state instead of the
    /// note copying the old state forward, which is how the summary lost
    /// every plot event one session after it happened.
    private func consolidate(_ context: MemorySessionContext) async {
        let scope = context.scope
        if unconsolidated[scope]?.session.id == context.session.id {
            unconsolidated[scope] = nil
        }
        guard let journal = await service.journalStore(for: scope) else { return }
        let turns = await journal.turns(session: context.session.id,
                                        limit: configuration.consolidationMaximumTurns,
                                        in: scope)
        guard !turns.isEmpty else { return }
        let existing = await service.recordedKeys(in: scope)
        let request = ServerMemory.consolidationRequest(
            turns: Array(turns.reversed()), existingKeys: existing, workspace: scope.workspace)
        let started = Date()
        let completion: ServerCompletion
        do {
            completion = try await inner.generate(request, onEvent: { _ in })
        } catch {
            ServerLog.memory("consolidation failed session=\(context.session.id): \(error)")
            return
        }
        let records = ServerMemory.consolidationRecords(from: completion.content)
        let written = await service.storeConsolidation(records, in: context)
        ServerLog.memory("consolidated session=\(context.session.id) turns=\(turns.count) "
                         + "facts=\(written) keys=\(records.map(\.key.rawValue).joined(separator: ","))"
                         + " prompt=\(completion.usage.promptTokens) "
                         + "completion=\(completion.usage.completionTokens) "
                         + "seconds=\(Int(Date().timeIntervalSince(started)))")
    }

    /// Identifies a conversation for the purpose of freezing its prompt and
    /// counting its turns.
    private func conversationKey(for request: ValidatedChatRequest) -> String {
        let placement = resolvePlacement(for: request)
        return ServerMemory.sessionIdentifier(messages: request.messages,
                                              workspace: placement.workspace)
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
        let placement = resolvePlacement(for: request)
        let id = ServerMemory.sessionIdentifier(messages: request.messages,
                                                workspace: placement.workspace)
        if let existing = contexts[id] { return existing }
        guard let context = await service.beginSession(
            id: id,
            workspaceOverride: placement.override,
            modelID: nil,
            tag: placement.tag) else { return nil }
        contexts[id] = context
        // A new session in a scope whose last session still has undistilled
        // turns is a rollover: the end-of-conversation signal the API never
        // sends. The previous session is consolidated once this request has
        // returned, so the person waiting on it does not pay for it.
        if configuration.sessionConsolidation,
           let previous = unconsolidated[context.scope],
           previous.session.id != context.session.id {
            idleTimers[context.scope]?.cancel()
            pendingAfterTurn[context.scope] = previous
            unconsolidated[context.scope] = nil
        }
        ServerLog.memory("session=\(context.session.id) scope=\(context.scope.workspace) "
                         + "tag=\(placement.tag ?? "-") via=\(placement.source) "
                         + "bootstrap=\(context.bootstrap.records.count) "
                         + "durable=\(context.isDurable)")
        return context
    }

    /// Where a conversation's memory lives, and why.
    private struct Placement {
        /// The workspace the session is placed in.
        let workspace: String
        /// The override handed to the service; nil means the launch workspace.
        let override: String?
        /// The label recorded on the session.
        let tag: String?
        /// For the log: "header", "declared-cwd" or "launch".
        let source: String
    }

    /// Decides the workspace for a request, in this order:
    ///
    /// 1. The `X-NVMAI-Workspace` header, when the client sent one.
    /// 2. The working directory the client declared in its system prompt.
    ///    This is the one that keeps a novel and a codebase apart with no
    ///    configuration at all: the coding CLIs already say where they are
    ///    on every request, and where they are is the project.
    /// 3. The launch directory.
    ///
    /// A declared directory that is not a project -- the home directory,
    /// the root -- falls through to the launch workspace and is logged once,
    /// rather than being refused: refusing a request over a client's cwd
    /// would turn a memory nicety into a serving failure.
    private func resolvePlacement(for request: ValidatedChatRequest) -> Placement {
        if let header = request.workspace {
            return Placement(workspace: header, override: header, tag: header, source: "header")
        }
        guard configuration.allowsPerRequestWorkspace,
              let declared = ServerMemory.declaredWorkingDirectory(in: request.messages)
        else {
            return Placement(workspace: configuration.workspace, override: nil,
                             tag: nil, source: "launch")
        }
        if let reason = MemoryConfiguration.junkDrawerReason(
            forPath: declared, environment: ["HOME": homeDirectory]) {
            if refusedDirectories.insert(declared).inserted {
                ServerLog.memory("declared working directory ignored: \(reason)")
            }
            return Placement(workspace: configuration.workspace, override: nil,
                             tag: nil, source: "launch")
        }
        let workspace = MemoryConfiguration.workspaceIdentifier(forPath: declared)
        let tag = URL(fileURLWithPath: declared).lastPathComponent
        return Placement(workspace: workspace, override: workspace, tag: tag,
                         source: "declared-cwd")
    }

    /// When the round limit stops a conversation mid-memory, say so in the
    /// finish reason rather than presenting a truncated answer as complete.
}

public extension MemoryBackend {
    /// Flush memory to disk and release the workspace locks.
    ///
    /// Called on the way out of a graceful shutdown. Session boundaries are
    /// the usual durability point, but a server told to stop mid-conversation
    /// has records that have not reached a barrier yet, and those are the
    /// ones a person would most notice losing.
    func shutDown() async {
        for timer in idleTimers.values { timer.cancel() }
        idleTimers.removeAll()
        await service.shutDown()
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
        guard configuration.isEnabled else {
            if let reason = configuration.disabledReason { ServerLog.memory(reason) }
            return backend
        }
        let service = MemoryService(configuration: configuration) { event in
            ServerLog.memory(event.message)
        }
        ServerLog.memory(configuration.summary)
        // Replay the workspace journal now, at boot, rather than when the
        // first request arrives and the model is about to need the disk.
        Task(priority: .utility) { await service.warmUp() }
        return MemoryBackend(wrapping: backend,
                             service: service,
                             configuration: configuration)
    }
}
