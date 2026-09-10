import Foundation
import NVMAI

/// Serving a small model from the CPU, through the same HTTP surface as the
/// big ones.
///
/// The side-engine began as a way to run a 2B beside a 35B for the memory
/// work, and it stays that. But a model that answers correctly at twenty
/// tokens a second is a model worth serving, and there is no reason a person
/// with a small model and no GPU budget should get a different API, a
/// different tokenizer path or a different set of options.
///
/// So this is a `ServerInferenceBackend` like `ServerModelSession`, and
/// everything above it — the OpenAI and Responses surfaces, the memory
/// subsystem, the watchdogs — works unchanged.
///
/// **What it does not do, deliberately.** No prompt cache: the KV state is
/// rebuilt per request, because a CPU engine's prefill is cheap relative to
/// its decode and the cache's complexity buys little. No expert streaming:
/// these models are dense. No MTP.
public actor CPUModelBackend: ServerInferenceBackend {

    private let model: CPUQwen35
    private let tokenizer: GFTokenizer
    private let context: Int
    private let defaults: GenerationDefaults.Sampling

    public nonisolated let residentBytes: Int
    /// The thread width in force, so the startup banner can report what the
    /// engine will actually use without importing the kernel.
    public nonisolated let threads: Int
    public nonisolated var maximumContext: Int { context }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { defaults }

    /// Loads a snapshot and, unless told otherwise, makes it resident.
    ///
    /// Residency is the point of the CPU path. Left to the page cache a
    /// side model is evicted by whatever else wants the memory, and the next
    /// request pays to fault the whole thing back in — measured on this
    /// project, an unhelpful `madvise` hint alone cost 2.6x throughput. A
    /// model asked for by name should be in memory.
    /// What the CPU path will serve however long the checkpoint says it can.
    ///
    /// Qwen3.5 claims 262,144 positions and the GPU engine honours it. Here
    /// attention is a loop over the cache and the cache is held per token,
    /// so a context that is merely large on a GPU is unusable on four cores:
    /// at 262k the key/value cache alone is over three gigabytes, and every
    /// token would walk all of it. A ceiling that is quietly enforced beats
    /// a promise that is quietly broken.
    public static let contextCeiling = 32_768

    /// `thinkingMode` is baked into the tokenizer's generation prompt, as it
    /// is on the GPU path: the template renders the thinking switch, so it is
    /// a load-time setting, not a per-request one.
    public init(snapshotDirectory: URL,
                maximumContext: Int = CPUModelBackend.contextCeiling,
                resident: Bool = true,
                thinkingMode: ModelThinkingMode = .off) async throws {
        let snapshot = try AffineSnapshot(directory: snapshotDirectory)
        guard let family = snapshot.family else {
            throw CPUBackendError.unsupported(
                CPUModelFamily.refusal(modelType: snapshot.modelType))
        }
        residentBytes = resident ? snapshot.makeResident() : 0
        let engine = try CPUQwen35(snapshot: snapshot)
        threads = engine.threads
        model = engine
        tokenizer = try await GFTokenizer.load(from: snapshotDirectory,
                                               thinkingMode: thinkingMode)
        context = min(maximumContext, snapshot.configuration.maxPositions,
                      Self.contextCeiling)
        // The family's own, which is what the catalog advertises for it, so
        // a launcher showing the defaults shows what a request will get.
        defaults = family.samplingDefaults
    }

    public enum CPUBackendError: Error, CustomStringConvertible {
        case unsupported(String)
        case promptTooLong(Int, Int)

        public var description: String {
            switch self {
            case .unsupported(let detail): detail
            case .promptTooLong(let count, let limit):
                "prompt is \(count) tokens and the context is \(limit)"
            }
        }
    }

    /// The thread width, which the caller sets from whether anyone is
    /// waiting on the GPU. One thread costs a concurrent 35B generation 3%
    /// and four costs 31%, measured, so this is not a detail.
    public func setContention(_ contention: (@Sendable () -> Bool)?) {
        model.contention = contention
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let rendered = try tokenizer.applyChatTemplate(request.messages)
        let promptIDs = tokenizer.encode(rendered, addBOS: false)
        let prompt = promptIDs.map(Int.init)
        guard prompt.count < context else {
            throw CPUBackendError.promptTooLong(prompt.count, context)
        }
        let budget = min(request.maximumCompletionTokens, context - prompt.count)
        let configuration = request.generationConfig
        let sampler = CPUSampler(
            temperature: configuration.temperature,
            topP: configuration.topP ?? 1,
            topK: configuration.topK ?? 0,
            // Deterministic unless the request asks for variety, so the same
            // prompt gives the same answer and a regression is visible.
            seed: configuration.temperature > 0 ? nil : 0)
        let generator = sampler.makeGenerator()

        model.reset()
        var logits: [Float] = []
        for (index, token) in prompt.enumerated() {
            logits = try model.step(token: token, needsLogits: index == prompt.count - 1)
        }

        // The same decoder the GPU path runs, so a thought is split the same
        // way on either engine. This path renders no tool template, so the
        // decoder only ever splits thoughts.
        let decoder = StructuredAssistantDecoder.forGeneration(
            tokenizer: tokenizer, promptIDs: promptIDs, allowedTools: nil)
        var detokenizer = GFDetokenizer(tokenizer: tokenizer)
        var output = AssistantOutput(stops: configuration.stopStrings, onEvent: onEvent)
        var produced = 0
        var reason = "length"
        while produced < budget {
            let next = sampler.pick(logits, using: generator)
            if next == Int(tokenizer.eosID) { reason = "stop"; break }
            produced += 1
            output.publish(try events(for: Int32(next), decoder: decoder,
                                      detokenizer: &detokenizer))
            if output.isStopped { reason = "stop"; break }
            if produced >= budget { break }
            logits = try model.step(token: next)
        }
        if let decoder {
            output.publish(try decoder.consumeTail(detokenizer.flush()))
            try decoder.finish()
        }
        output.finish()
        return ServerCompletion(
            content: output.content,
            toolCalls: [],
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: prompt.count,
                               completionTokens: produced,
                               totalTokens: prompt.count + produced,
                               cachedTokens: 0),
            reasoning: output.reasoning)
    }

    /// One sampled token as decoder events.
    ///
    /// With no decoder -- thinking off -- this is the per-token decode the
    /// CPU path has always done, so that output does not move by a byte. A
    /// thinking model needs the streaming detokenizer instead: the decoder
    /// knows `<think>` and `</think>` by their literal text on the delta, and
    /// a per-token decode that skips special tokens drops exactly those.
    private func events(for token: Int32,
                        decoder: StructuredAssistantDecoder?,
                        detokenizer: inout GFDetokenizer) throws -> [StructuredAssistantEvent] {
        guard let decoder else {
            let piece = tokenizer.decode([token], skipSpecialTokens: true)
            return piece.isEmpty ? [] : [.content(piece)]
        }
        return try decoder.consume(tokenID: token, delta: detokenizer.push(token))
    }
}

/// The Messages API's count_tokens, from the same rendering `generate` uses,
/// so a client sizing its context against a CPU model gets the real number.
extension CPUModelBackend: PromptTokenCounting {
    public func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        try Self.promptTokenCount(request, tokenizer: tokenizer)
    }

    /// The count from a tokenizer alone, which is how the router answers for
    /// a CPU model that is not the one loaded.
    static func promptTokenCount(_ request: ValidatedChatRequest,
                                 tokenizer: GFTokenizer) throws -> Int {
        let rendered = try tokenizer.applyChatTemplate(request.messages)
        return tokenizer.encode(rendered, addBOS: false).count
    }
}
