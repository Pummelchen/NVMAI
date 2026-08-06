import CryptoKit
import Foundation
import NVMAI

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

// MARK: - Structured Output Diagnostics (#90)

/// Kinds of structured-output failures that can be diagnosed.
enum StructuredOutputFailureKind: String, Equatable, Sendable {
    case decoderConsume = "decoder_consume"
    case decoderFinish = "decoder_finish"
    case orphanToolResponse = "orphan_tool_response"
}

/// Classifies the root cause of a structured-output failure.
enum StructuredOutputFailureCause: String, Equatable, Sendable {
    case malformed
    case unknownTool = "unknown_tool"
    case oversized
    case unexpected
    case none

    static func classify(_ error: Error) -> Self {
        guard let parserError = error as? ToolCallParserError else {
            return .unexpected
        }
        switch parserError {
        case .malformed: return .malformed
        case .unknownTool: return .unknownTool
        case .oversized: return .oversized
        }
    }
}

/// Rich diagnostic snapshot collected at structured-output failure time.
/// Includes SHA-256 hashes of token sequences for forensic comparison.
struct StructuredOutputFailureDiagnostics: Equatable, Sendable {
    let renderedPromptTokens: Int
    let effectivePromptTokens: Int
    let resultPromptTokens: Int
    let cachedPromptTokens: Int
    let computedPrefillTokens: Int
    let completionTokens: Int
    let maxCompletionTokens: Int
    let rawStop: String
    let kvPosition: Int
    let kvBackedTokens: Int
    let boundaryTokens: Int
    let decodedCalls: Int
    let visibleBytes: Int
    let stopStringMatched: Bool
    let toolStartCount: Int
    let toolEndCount: Int
    let toolResponseCount: Int
    let toolResponseEndCount: Int
    let lastToolStartOffset: Int
    let lastToolEndOffset: Int
    let lastToolResponseOffset: Int
    let lastToolResponseEndOffset: Int
    let effectiveCountMatchesResult: Bool
    let effectivePrefixMatchesKV: Bool
    let kvPositionMatchesHistory: Bool
    let completionCountMatchesHistory: Bool
    let prefillAccountingMatches: Bool
    let renderedPromptHash: String
    let effectivePromptHash: String
    let generatedHash: String

    init(
        renderedPromptIDs: [Int32],
        effectivePromptIDs: [Int32],
        result: RawDecodeResult,
        maxCompletionTokens: Int,
        decodedCalls: Int,
        visibleBytes: Int,
        stopStringMatched: Bool,
        toolStartID: Int32,
        toolEndID: Int32,
        toolResponseID: Int32,
        toolResponseEndID: Int32
    ) {
        let safePrefillCount = min(
            max(result.prefillTokens, 0),
            result.kvBackedTokenIDs.count)
        let committedGenerated = result.kvBackedTokenIDs.dropFirst(safePrefillCount)
        let boundary = result.uncommittedBoundaryTokenIDs[...]
        let generatedSegments = [committedGenerated, boundary]

        var toolStartCount = 0
        var toolEndCount = 0
        var toolResponseCount = 0
        var toolResponseEndCount = 0
        var lastToolStartOffset = -1
        var lastToolEndOffset = -1
        var lastToolResponseOffset = -1
        var lastToolResponseEndOffset = -1
        var offset = 0
        for segment in generatedSegments {
            for tokenID in segment {
                if tokenID == toolStartID {
                    toolStartCount += 1
                    lastToolStartOffset = offset
                }
                if tokenID == toolEndID {
                    toolEndCount += 1
                    lastToolEndOffset = offset
                }
                if tokenID == toolResponseID {
                    toolResponseCount += 1
                    lastToolResponseOffset = offset
                }
                if tokenID == toolResponseEndID {
                    toolResponseEndCount += 1
                    lastToolResponseEndOffset = offset
                }
                offset += 1
            }
        }

        let (prefillAccounted, prefillOverflow) = result.cachedPromptTokens
            .addingReportingOverflow(result.computedPrefillTokens)

        self.renderedPromptTokens = renderedPromptIDs.count
        self.effectivePromptTokens = effectivePromptIDs.count
        self.resultPromptTokens = result.prefillTokens
        self.cachedPromptTokens = result.cachedPromptTokens
        self.computedPrefillTokens = result.computedPrefillTokens
        self.completionTokens = result.newTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.rawStop = Self.rawStop(result.reason)
        self.kvPosition = result.kvPosition
        self.kvBackedTokens = result.kvBackedTokenIDs.count
        self.boundaryTokens = result.uncommittedBoundaryTokenIDs.count
        self.decodedCalls = decodedCalls
        self.visibleBytes = visibleBytes
        self.stopStringMatched = stopStringMatched
        self.toolStartCount = toolStartCount
        self.toolEndCount = toolEndCount
        self.toolResponseCount = toolResponseCount
        self.toolResponseEndCount = toolResponseEndCount
        self.lastToolStartOffset = lastToolStartOffset
        self.lastToolEndOffset = lastToolEndOffset
        self.lastToolResponseOffset = lastToolResponseOffset
        self.lastToolResponseEndOffset = lastToolResponseEndOffset
        self.effectiveCountMatchesResult = effectivePromptIDs.count == result.prefillTokens
        self.effectivePrefixMatchesKV = result.kvBackedTokenIDs.count >= effectivePromptIDs.count
            && result.kvBackedTokenIDs.prefix(effectivePromptIDs.count)
                .elementsEqual(effectivePromptIDs)
        self.kvPositionMatchesHistory = result.kvPosition == result.kvBackedTokenIDs.count
        self.completionCountMatchesHistory = offset == result.newTokens
        self.prefillAccountingMatches = !prefillOverflow
            && prefillAccounted == result.prefillTokens
        self.renderedPromptHash = Self.i32leSHA256([renderedPromptIDs[...]])
        self.effectivePromptHash = Self.i32leSHA256([effectivePromptIDs[...]])
        self.generatedHash = Self.i32leSHA256(generatedSegments)
    }

    /// SHA-256 over little-endian UInt32 byte representations of token IDs.
    static func i32leSHA256(_ segments: [ArraySlice<Int32>]) -> String {
        var hasher = SHA256()
        var bytes = Data()
        bytes.reserveCapacity(4_096)
        for segment in segments {
            for tokenID in segment {
                var littleEndian = UInt32(bitPattern: tokenID).littleEndian
                withUnsafeBytes(of: &littleEndian) {
                    bytes.append(contentsOf: $0)
                }
                if bytes.count == 4_096 {
                    hasher.update(data: bytes)
                    bytes.removeAll(keepingCapacity: true)
                }
            }
        }
        if !bytes.isEmpty { hasher.update(data: bytes) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    var logDescription: String {
        [
            "rendered_prompt_tokens=\(renderedPromptTokens)",
            "effective_prompt_tokens=\(effectivePromptTokens)",
            "result_prompt_tokens=\(resultPromptTokens)",
            "cached_prompt_tokens=\(cachedPromptTokens)",
            "computed_prefill_tokens=\(computedPrefillTokens)",
            "completion_tokens=\(completionTokens)",
            "max_completion_tokens=\(maxCompletionTokens)",
            "raw_stop=\(rawStop)",
            "kv_position=\(kvPosition)",
            "kv_backed_tokens=\(kvBackedTokens)",
            "boundary_tokens=\(boundaryTokens)",
            "decoded_calls=\(decodedCalls)",
            "visible_bytes=\(visibleBytes)",
            "stop_string_matched=\(stopStringMatched)",
            "tool_start_count=\(toolStartCount)",
            "tool_end_count=\(toolEndCount)",
            "tool_response_count=\(toolResponseCount)",
            "tool_response_end_count=\(toolResponseEndCount)",
            "last_tool_start_offset=\(lastToolStartOffset)",
            "last_tool_end_offset=\(lastToolEndOffset)",
            "last_tool_response_offset=\(lastToolResponseOffset)",
            "last_tool_response_end_offset=\(lastToolResponseEndOffset)",
            "effective_count_matches_result=\(effectiveCountMatchesResult)",
            "effective_prefix_matches_kv=\(effectivePrefixMatchesKV)",
            "kv_position_matches_history=\(kvPositionMatchesHistory)",
            "completion_count_matches_history=\(completionCountMatchesHistory)",
            "prefill_accounting_matches=\(prefillAccountingMatches)",
            "rendered_prompt_i32le_sha256=\(renderedPromptHash)",
            "effective_prompt_i32le_sha256=\(effectivePromptHash)",
            "generated_i32le_sha256=\(generatedHash)",
        ].joined(separator: " ")
    }

    private static func rawStop(_ reason: StopReason) -> String {
        switch reason {
        case .eos: "eos"
        case .endOfTurn: "end_of_turn"
        case .maxTokens: "max_tokens"
        case .stopString: "stop_string"
        case .toolCalls: "tool_calls"
        case .external: "external"
        }
    }
}

/// A structured-output failure with full diagnostic context.
struct StructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let kind: StructuredOutputFailureKind
    let cause: StructuredOutputFailureCause
    let diagnostics: StructuredOutputFailureDiagnostics

    var debugDescription: String {
        "structured_output_failure kind=\(kind.rawValue) "
            + "cause=\(cause.rawValue) \(diagnostics.logDescription)"
    }
}

// MARK: - End of Structured Output Diagnostics

public protocol ServerInferenceBackend: Sendable {
    /// The backend's configured context window, used to validate
    /// max_tokens/max_completion_tokens against the session's maxContext (S11).
    var maximumContext: Int { get }
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public extension ServerInferenceBackend {
    var maximumContext: Int {
        RuntimeConfiguration.supportedContextTokens.max() ?? 262_144
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var admittedCount = 0
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
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        // S6: strictly fewer than queueLimit admitted requests; `<=` admitted
        // queueLimit + 1.
        guard admittedCount < queueLimit else { throw ServerRequestError.queueFull }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation(prepared)
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
    /// Family-derived API model identifier used when --model-id is absent.
    public nonisolated var defaultModelID: String {
        switch modelFamily {
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen36MTP: return "qwen3.6-35b-a3b-mtp"
        }
    }
    /// The session's configured context window; the HTTP layer validates
    /// max_tokens against it (S11).
    public nonisolated var maximumContext: Int { maxContext }
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
        let loadRuntime = try RuntimeConfiguration(forceLogitsHead: true)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            expecting: .qwen36_35B_A3B,
            streamingMode: .pread(slotCount: loadRuntime.expertCacheSlots),
            expertCachePolicy: loadRuntime.modelExpertCachePolicy,
            integrityPolicy: .resolved(directoryURL: modelDirectory))
        let runtime = try RuntimeConfiguration(
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
                integrityPolicy: .resolved(directoryURL: mtpModelDirectory))
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
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        let effectivePromptIDs: [Int32]
        var completionStart: RawCompletionStart
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
                if runner.continuationPosition != cached {
                    // S15: the live KV no longer sits at the cached entry's
                    // position; re-prefill instead of resuming from a stale
                    // or mismatched in-memory state.
                    promptCache.invalidate()
                    effectivePromptIDs = promptIDs
                    completionStart = .reset
                } else {
                    effectivePromptIDs = effective
                    completionStart = .resume(cachedPromptTokens: cached)
                }
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
                if entryID == activePromptCacheEntryID,
                   runner.continuationPosition == cached {
                    // S15: tier=live is only trusted when the in-memory KV
                    // still matches the entry (same entry id and the KV
                    // cursor sits exactly at the request's expected
                    // position). Anything else falls through to a snapshot
                    // restore or a full prefill instead of resuming from a
                    // stale or mismatched KV.
                    print(
                        "NVMAI prompt_cache hit tier=live "
                            + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())")
                } else {
                    do {
                        guard let promptStateStore else {
                            throw ServerPromptStateStoreError.missing(entryID)
                        }
                        let tier = try await promptStateStore.restore(
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
        // S12: an identical-prompt replay whose render equals the entry's
        // KV-backed prefix has nothing to prefill (cached == prompt count).
        // The continuation API requires cached < prompt count (it must
        // prefill at least one token), so resume as a full prefill; the
        // entry stays active for later extending requests.
        if case .resume(let cached) = completionStart,
           cached >= effectivePromptIDs.count {
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
        func structuredFailure(
            kind: StructuredOutputFailureKind,
            cause: StructuredOutputFailureCause
        ) -> StructuredOutputFailure {
            StructuredOutputFailure(
                kind: kind,
                cause: cause,
                diagnostics: StructuredOutputFailureDiagnostics(
                    renderedPromptIDs: promptIDs,
                    effectivePromptIDs: effectivePromptIDs,
                    result: result,
                    maxCompletionTokens: config.maxNewTokens,
                    decodedCalls: calls.count,
                    visibleBytes: content.utf8.count,
                    stopStringMatched: stopMatcher.isStopped,
                    toolStartID: tokenizer.toolCallStartID,
                    toolEndID: tokenizer.toolCallEndID,
                    toolResponseID: tokenizer.toolResponseID,
                    toolResponseEndID: tokenizer.toolResponseEndID))
        }
        if let decodingError {
            throw structuredFailure(
                kind: .decoderConsume,
                cause: .classify(decodingError))
        }
        do {
            try decoder?.finish()
        } catch {
            throw structuredFailure(
                kind: .decoderFinish,
                cause: .classify(error))
        }
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw structuredFailure(kind: .orphanToolResponse, cause: .none)
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
                    // S2: capture is bounded by the store's hard snapshot cap;
                    // the payload is a plain Data copy, so the disk write can
                    // proceed off the actor (dedicated store disk queue) while
                    // the next request starts. Concurrent saves serialize on
                    // the queue, so a later generation's snapshot can never
                    // clobber an in-flight write. The entry is already in the
                    // in-memory cache; a request that races the write simply
                    // misses and re-prefills (restore failure self-heals).
                    let snapshot = try runner.captureInferenceState(
                        maximumBytes: promptStateStore.maximumSnapshotBytes)
                    guard snapshot.descriptor.position == publication.entry.kvPosition else {
                        throw InferenceStateSnapshotError.invalidPosition(
                            snapshot.descriptor.position)
                    }
                    let entry = publication.entry
                    Task.detached(priority: .utility) { [promptStateStore] in
                        let saved = await promptStateStore.save(
                            entry: entry,
                            snapshot: snapshot)
                        if let diskError = saved.diskError {
                            print("NVMAI prompt_cache disk_write_failed error=\(diskError)")
                        }
                        print(
                            "NVMAI prompt_cache stored "
                                + "tokens=\(entry.kvPosition) "
                                + "state_bytes=\(snapshot.payload.count) "
                                + "ram_bytes=\(saved.memoryBytes) "
                                + "disk_bytes=\(saved.diskBytes) "
                                + "entry=\(entry.id.uuidString.lowercased())")
                    }
                } catch {
                    // S24: a snapshot that cannot be captured or verified is
                    // never left published without backing; drop the entry so
                    // the next hit re-prefills instead of a doomed restore.
                    print("NVMAI prompt_cache snapshot_failed error=\(error)")
                    promptCache.remove(entryIDs: [publication.entry.id])
                    activePromptCacheEntryID = nil
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
            // S26: completion_tokens reports the number of GENERATED tokens,
            // matching OpenAI's "completion_tokens = tokens in the generated
            // completion". A stop-string-hidden suffix is therefore counted as
            // generated even though it is filtered from the visible content.
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
