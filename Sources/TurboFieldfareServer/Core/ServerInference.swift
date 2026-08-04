import CryptoKit
import Foundation
import TurboFieldfare

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage

    public init(content: String,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage) {
        self.content = content
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
    }
}

public protocol ServerInferenceBackend: Sendable {
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var active = false
    private var waiters: [Waiter] = []
    private var shuttingDown = false

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation()
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

public actor ServerModelSession: ServerInferenceBackend {
    /// Chat dialect of the loaded tokenizer; drives request-validation rules.
    public nonisolated let chatDialect: ChatDialect
    /// Family-derived API model identifier used when --model-id is absent.
    public nonisolated var defaultModelID: String {
        switch modelFamily {
        case .gemma4: return "gemma-4-26b-a4b-it"
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen36MTP: return "qwen3.6-35b-a3b-mtp"
        }
    }
    private nonisolated let modelFamily: ModelFamily

    private let context: MetalContext
    private let model: Model
    private let tokenizer: GFTokenizer
    private let runner: RealForwardRunner
    private let mtpDecoder: StreamingMTPDecoder?
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    public nonisolated let prefillChunkTokens: Int
    private let maxContext: Int
    public nonisolated let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache: ServerPromptCache
    private let promptStateStore: ServerPromptStateStore?
    private var activePromptCacheEntryID: UUID?

    static func effectivePromptCacheMode(
        requested: ServerPromptCacheMode,
        mtpEnabled: Bool
    ) -> ServerPromptCacheMode {
        // A target-only snapshot cannot restore the draft stream. Keeping a
        // cache allocated while MTP is active would spend memory on entries
        // that must never be consumed or published.
        mtpEnabled ? .off : requested
    }

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            promptCacheMode: ServerPromptCacheMode = .multiPrefix,
                            promptCacheMaximumEntries: Int = 4,
                            promptCacheMemoryLimitBytes: Int = 256 * 1_048_576,
                            promptCacheDiskDirectory: URL? = nil,
                            promptCacheDiskLimitBytes: Int = 8_192 * 1_048_576,
                            prefillChunkTokens requestedPrefillChunkTokens: Int? = nil,
                            mtpModelDirectory: URL? = nil,
                            mtpMemoryMiB: Int = StreamingMTPMemoryPlan.defaultBudgetMiB) async throws -> ServerModelSession {
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        let tokenizer = try await GFTokenizer.load(from: tokenizerFolder)
        let context = try MetalContext()
        let loadRuntime = RuntimeConfiguration(forceLogitsHead: true)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: .pread(slotCount: loadRuntime.expertCacheSlots),
            expertCachePolicy: loadRuntime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runtime = RuntimeConfiguration(
            expertCacheSlots: loadRuntime.expertCacheSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: loadRuntime.rdadvisePolicy,
            prefillChunkTokens: requestedPrefillChunkTokens
                ?? (model.config.family == .qwen36
                    ? RuntimeConfiguration.qwenLongPrefillChunkTokens
                    : loadRuntime.prefillChunkTokens),
            prefillAttentionPath: loadRuntime.prefillAttentionPath,
            forceLogitsHead: true)
        let mtpDecoder: StreamingMTPDecoder?
        let runner: RealForwardRunner
        if let mtpModelDirectory {
            let sidecar = try Model.load(
                directoryURL: mtpModelDirectory,
                device: context.device,
                expecting: .qwen36MTP,
                streamingMode: .pread(slotCount: StreamingMTPMemoryPlan.expertSlots),
                expertCachePolicy: runtime.modelExpertCachePolicy,
                integrityPolicy: .fullSha256)
            let decoder = try StreamingMTPDecoder(
                targetModel: model,
                mtpSidecar: sidecar,
                context: context,
                maxContext: maxContext,
                memoryBudgetMiB: mtpMemoryMiB,
                runtimeConfiguration: runtime)
            mtpDecoder = decoder
            runner = decoder.target
        } else {
            mtpDecoder = nil
            runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext,
                                           runtimeConfiguration: runtime)
        }
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: PrefillKVStorageMode.fp16.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let effectivePromptCacheMode = Self.effectivePromptCacheMode(
            requested: promptCacheMode,
            mtpEnabled: mtpDecoder != nil)
        let promptStateStore: ServerPromptStateStore?
        let promptCache: ServerPromptCache
        if effectivePromptCacheMode == .multiPrefix {
            let store = try ServerPromptStateStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: promptCacheMemoryLimitBytes,
                    diskDirectory: promptCacheDiskDirectory,
                    diskLimitBytes: promptCacheDiskLimitBytes))
            let persisted = store.loadEntries(domain: promptCacheDomain)
            if persisted.count > promptCacheMaximumEntries {
                store.remove(entryIDs: persisted
                    .dropLast(promptCacheMaximumEntries)
                    .map(\.id))
            }
            promptStateStore = store
            promptCache = ServerPromptCache(
                maximumEntries: promptCacheMaximumEntries,
                entries: persisted)
        } else {
            promptStateStore = nil
            promptCache = ServerPromptCache(maximumEntries: 1)
        }
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  runner: runner,
                                  mtpDecoder: mtpDecoder,
                                  scratch: scratch,
                                  prefillConfig: runtime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: effectivePromptCacheMode,
                                  promptCacheDomain: promptCacheDomain,
                                  promptCache: promptCache,
                                  promptStateStore: promptStateStore)
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: GFTokenizer,
                 runner: RealForwardRunner,
                 mtpDecoder: StreamingMTPDecoder?,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain,
                 promptCache: ServerPromptCache,
                 promptStateStore: ServerPromptStateStore?) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.chatDialect = tokenizer.dialect
        self.modelFamily = model.config.family
        self.runner = runner
        self.mtpDecoder = mtpDecoder
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.prefillChunkTokens = prefillConfig.chunkTokens
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
        self.promptCache = promptCache
        self.promptStateStore = promptStateStore
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        var completed = false
        defer {
            if !completed {
                if promptCacheMode == .singlePrefix {
                    promptCache.invalidate()
                }
                activePromptCacheEntryID = nil
                runner.reset()
                mtpDecoder?.reset()
            }
        }
        let needsToolTemplate = usesToolTemplate(
            messages: request.messages,
            tools: request.tools)
        let promptIDs = try encodePrompt(
            messages: request.messages,
            tools: request.tools,
            usesToolTemplate: needsToolTemplate)
        if let audit = request.filterAudit {
            let originalPromptIDs = try encodePrompt(
                messages: audit.originalMessages,
                tools: audit.originalTools,
                usesToolTemplate: usesToolTemplate(
                    messages: audit.originalMessages,
                    tools: audit.originalTools))
            print(
                "NVMAI OpenCode profile=\(audit.profile.rawValue) "
                    + "body_bytes=\(audit.originalBodyBytes) "
                    + "prompt_tokens=\(originalPromptIDs.count)->\(promptIDs.count) "
                    + "messages=\(audit.originalMessages.count)->\(request.messages.count) "
                    + "tools=\(audit.originalTools.count)->\(request.tools.count)")
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer) {
            case .miss:
                promptCache.invalidate()
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(_, let effective, let cached):
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else if promptCacheMode == .multiPrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer) {
            case .miss:
                activePromptCacheEntryID = nil
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(let entryID, let effective, let cached):
                if entryID != activePromptCacheEntryID {
                    do {
                        guard let promptStateStore else {
                            throw ServerPromptStateStoreError.missing(entryID)
                        }
                        let tier = try promptStateStore.restore(
                            entryID: entryID,
                            into: runner)
                        print(
                            "NVMAI prompt_cache hit tier=\(tier) "
                                + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())")
                    } catch {
                        print(
                            "NVMAI prompt_cache restore_failed "
                                + "entry=\(entryID.uuidString.lowercased()) error=\(error)")
                        promptStateStore?.remove(entryIDs: [entryID])
                        promptCache.remove(entryIDs: [entryID])
                        activePromptCacheEntryID = nil
                        effectivePromptIDs = promptIDs
                        completionStart = .reset
                        break
                    }
                } else {
                    print(
                        "NVMAI prompt_cache hit tier=live "
                            + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())")
                }
                activePromptCacheEntryID = entryID
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else {
            promptCache.invalidate()
            activePromptCacheEntryID = nil
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        let decoder = needsToolTemplate
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)))
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false

        let activeProducer: any LogitProducer = if config.isPureGreedy,
                                                   let mtpDecoder,
                                                   promptIDs.count + config.maxNewTokens
                                                    <= mtpDecoder.draftMaxContext {
            mtpDecoder
        } else {
            runner
        }
        let activeStart: RawCompletionStart = activeProducer is StreamingMTPDecoder
            ? .reset : completionStart
        let activePromptIDs = activeProducer is StreamingMTPDecoder
            ? promptIDs : effectivePromptIDs
        let result = try await runRawCompletion(
            producer: activeProducer,
            tokenizer: tokenizer,
            promptIds: activePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: activeStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events = if let decoder {
                            try decoder.consume(tokenID: tokenID, delta: delta)
                        } else {
                            delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        for event in events {
                            switch event {
                            case .content(let text):
                                let visible = stopMatcher.push(text)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    case .tail(let text):
                        let visible = stopMatcher.push(text)
                        if !visible.isEmpty {
                            content += visible
                            onEvent(.content(visible))
                        }
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        if let activeMTP = activeProducer as? StreamingMTPDecoder {
            let stats = activeMTP.statistics
            let decodeRate = result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            print(String(format:
                "NVMAI mtp drafted=%d accepted=%d acceptance=%.1f%% "
                    + "target_passes=%d emitted_per_pass=%.3f "
                    + "prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f "
                    + "memory_required_mib=%.1f memory_budget_mib=%.1f",
                stats.draftedTokens,
                stats.acceptedTokens,
                stats.acceptanceRate * 100,
                stats.targetBackbonePasses,
                stats.emittedTokensPerTargetPass,
                result.prefillSeconds,
                result.decodeSeconds,
                decodeRate,
                Double(activeMTP.memoryPlan.requiredBytes) / 1_048_576,
                Double(activeMTP.memoryPlan.budgetBytes) / 1_048_576))
        } else {
            let decodeRate = result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            print(String(format:
                "NVMAI generation prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f",
                result.prefillSeconds,
                result.decodeSeconds,
                decodeRate))
        }
        if let decodingError { throw decodingError }
        try decoder?.finish()
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw GemmaToolCallParserError.malformed
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        if mtpDecoder != nil {
            // Native MTP keeps a second KV stream. Until both states are
            // persisted atomically, do not publish target-only cache entries.
            promptCache.invalidate()
            activePromptCacheEntryID = nil
        } else if promptCacheMode == .singlePrefix {
            let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopMatcher.isStopped)
            if publication == nil { promptCache.invalidate() }
        } else if promptCacheMode == .multiPrefix {
            let previousActive = activePromptCacheEntryID
            if let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopMatcher.isStopped) {
                promptStateStore?.remove(entryIDs: publication.evictedEntryIDs)
                do {
                    guard let promptStateStore else {
                        throw ServerPromptStateStoreError.missing(
                            publication.entry.id)
                    }
                    let snapshot = try runner.captureInferenceState(
                        maximumBytes: promptStateStore.maximumSnapshotBytes)
                    guard snapshot.descriptor.position == publication.entry.kvPosition else {
                        throw InferenceStateSnapshotError.invalidPosition(
                            snapshot.descriptor.position)
                    }
                    let saved = promptStateStore.save(
                        entry: publication.entry,
                        snapshot: snapshot)
                    let invalidated = saved.unbackedEntryIDs.filter {
                        $0 != publication.entry.id
                    }
                    promptCache.remove(entryIDs: invalidated)
                    if let diskError = saved.diskError {
                        print("NVMAI prompt_cache disk_write_failed error=\(diskError)")
                    }
                    print(
                        "NVMAI prompt_cache stored "
                            + "tokens=\(publication.entry.kvPosition) "
                            + "state_bytes=\(snapshot.payload.count) "
                            + "ram_bytes=\(saved.memoryBytes) "
                            + "disk_bytes=\(saved.diskBytes) "
                            + "entry=\(publication.entry.id.uuidString.lowercased())")
                } catch {
                    print("NVMAI prompt_cache snapshot_failed error=\(error)")
                }
                if let previousActive,
                   previousActive != publication.entry.id,
                   promptStateStore?.contains(previousActive) != true {
                    promptCache.remove(entryIDs: [previousActive])
                }
                activePromptCacheEntryID = publication.entry.id
            } else {
                activePromptCacheEntryID = nil
            }
        }
        completed = true
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens))
    }

    private func usesToolTemplate(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> Bool {
        !tools.isEmpty || messages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
    }

    private func encodePrompt(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool
    ) throws -> [Int32] {
        if usesToolTemplate {
            return try tokenizer.encodeToolChat(messages: messages, tools: tools)
        }
        let rendered = try tokenizer.applyChatTemplate(messages)
        return tokenizer.encode(rendered, addBOS: false)
    }
}
