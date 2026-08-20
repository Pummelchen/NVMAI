import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// unchecked-invariant: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int, logitSoftcap: Float = 0.0) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab,
                                   logitSoftcap: logitSoftcap)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && presencePenalty == 0 && repetitionPenalty == 1
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
/// lint:allow-long the generation loop: continuation validation, the prefill
/// mode switch, then token-by-token decode with stop matching and progress
/// reporting. The loop body reads and writes the same half-dozen pieces of
/// decode state on every iteration, so splitting it would thread that state
/// back through parameters on every call.
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             start: RawCompletionStart = .reset,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    if let mtp = producer as? StreamingMTPDecoder {
        return try await runStreamingMTPCompletion(
            decoder: mtp,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: start,
            shouldStop: shouldStop,
            onProgress: onProgress)
    }
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    let computedPrefillTokens = promptIds.count - cachedPromptTokens

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting {
        // A resume already occupies `cachedPromptTokens` KV rows, so only the
        // uncached prompt plus the response is new work — it must fit the
        // remaining capacity. Algebraically this is the final-KV-position
        // bound (`promptIds.count + maxNewTokens <= maxContext`); written in
        // remaining-capacity form so near-maxContext continuations are not
        // over-rejected (R9).
        let newRows = (promptIds.count - cachedPromptTokens) + config.maxNewTokens
        let remainingCapacity = context.maxContext - cachedPromptTokens
        if newRows > remainingCapacity {
            throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                                 maxNew: config.maxNewTokens,
                                                 maxContext: context.maxContext)
        }
    }
    switch start {
    case .reset:
        producer.reset()
    case .resume:
        // Re-derive the conformance rather than force-cast on the guard 30
        // lines above: a trap here would take down the server process, and the
        // invariant is far enough away to be broken by an unrelated edit.
        guard let continuable = producer as? any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[cachedPromptTokens...]
    switch prefillConfig.mode {
    case .chunked where producer is any ChunkedPrefillRunner:
        // lint:allow-force the `where` clause one line above is the guard; a
        // producer without the conformance falls through to plain `.chunked`.
        let chunked = producer as! any ChunkedPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: mode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: cachedPromptTokens + done, total: promptIds.count))
        }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case .chunked:
        throw PrefillError.chunkedUnsupported(
            PrefillError.chunkedRequiresChunkedRunnerReason)
    case .off:
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
    }

    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    // The scratch sampler persists across generations; its incremental
    // repetition-penalty history is per-generation (R25).
    scratch.sampler.resetPenaltyHistory()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated)
            }
        } else if fusedGreedy {
            tokenID = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else {
            tokenID = try sampleOnce(scratch: scratch, context: context,
                                 history: history, config: config, position: generated)
        }
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            if hitStopString {
                // A configured stop string truncates output; the caller's
                // external stop signal reports `.external` instead (R35).
                reason = stopMatcher.isStopped ? .stopString : .external
            } else {
                reason = .maxTokens
            }
            break
        }

        history.append(tokenID)
        try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs)
}

private func runStreamingMTPCompletion(
    decoder: StreamingMTPDecoder,
    tokenizer: GFTokenizer,
    promptIds: [Int32],
    config: GenerationConfig,
    scratch: RawCompletionScratch,
    prefillConfig: PrefillRuntimeConfig,
    start: RawCompletionStart,
    shouldStop: () -> Bool,
    onProgress: (RawDecodeProgress) -> Void
) async throws -> RawDecodeResult {
    try config.validate()
    guard config.isPureGreedy else { throw StreamingMTPError.greedyOnly }
    guard case .reset = start else {
        throw GeneratorError.invalidContinuation(
            "MTP continuation snapshots are not yet persisted; start a fresh request")
    }
    guard !promptIds.isEmpty else { throw GeneratorError.emptyPrompt }

    let prefillStart = Date()
    var boundary = try await decoder.prepare(
        promptIds: promptIds,
        config: config,
        prefillConfig: prefillConfig,
        logits: scratch.logits) { done in
            onProgress(.prefill(done: done, total: promptIds.count))
        }
    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var backedHistory = promptIds
    var uncommitted: [Int32] = []
    var pending: [(token: Int32, backed: Bool)] = [(boundary, false)]

    decodeLoop: while true {
        while !pending.isEmpty {
            try Task.checkCancellation()
            let item = pending.removeFirst()
            boundary = item.token
            generated += 1
            // Mirrors the scalar loop: `uncommitted` holds the last emitted
            // token iff advance has not yet committed it to the target KV
            // (R10). A token reported backed by the batch is already in the
            // KV, so it never sits uncommitted.
            uncommitted = item.backed ? [] : [item.token]

            if tokenizer.stopTokenIDs.contains(item.token)
                || config.extraStopTokens.contains(item.token) {
                if item.token == tokenizer.endOfTurnID { reason = .endOfTurn }
                else if item.token == tokenizer.toolResponseID { reason = .toolCalls }
                else { reason = .eos }
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                break decodeLoop
            }
            let visible = stopMatcher.push(detok.push(item.token))
            onProgress(.token(index: generated - 1, id: item.token, delta: visible))
            let hitStop = stopMatcher.isStopped || shouldStop()
            let hitMax = generated >= config.maxNewTokens
            if hitStop || hitMax {
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                if hitStop {
                    reason = stopMatcher.isStopped ? .stopString : .external
                } else {
                    reason = .maxTokens
                }
                break decodeLoop
            }
        }

        // The boundary is reported backed only after the advance that commits
        // it to the target KV succeeds (R10): "reported backed" strictly means
        // "committed by a completed advance", so the final boundary token is
        // always accounted for in `kvBackedTokenIDs`.
        let batch = try await decoder.advance(boundaryToken: boundary)
        backedHistory.append(boundary)
        pending = batch.tokenIDs.enumerated().map { index, token in
            (token, index < batch.backedPrefixCount)
        }
        if batch.backedPrefixCount > 0 {
            backedHistory.append(contentsOf: batch.tokenIDs.prefix(batch.backedPrefixCount))
        }
    }

    return RawDecodeResult(
        prefillTokens: promptIds.count,
        cachedPromptTokens: 0,
        computedPrefillTokens: promptIds.count,
        prefillSeconds: prefillSeconds,
        newTokens: generated,
        decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason,
        kvPosition: decoder.targetPosition,
        kvBackedTokenIDs: backedHistory,
        uncommittedBoundaryTokenIDs: uncommitted)
}

private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int) throws -> Int32 {
    guard let cb = context.queue.makeCommandBuffer() else {
        throw ModelError.residentBufferWrapFailed
    }
    try scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                               history: history, config: config, position: position,
                               outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}
