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

    public init(snapshotDirectory: URL,
                maximumContext: Int = CPUModelBackend.contextCeiling,
                resident: Bool = true) async throws {
        let snapshot = try AffineSnapshot(directory: snapshotDirectory)
        guard snapshot.family != nil else {
            throw CPUBackendError.unsupported(
                CPUModelFamily.refusal(modelType: snapshot.modelType))
        }
        residentBytes = resident ? snapshot.makeResident() : 0
        let engine = try CPUQwen35(snapshot: snapshot)
        threads = engine.threads
        model = engine
        tokenizer = try await GFTokenizer.load(from: snapshotDirectory)
        context = min(maximumContext, snapshot.configuration.maxPositions,
                      Self.contextCeiling)
        // The house settings: this family ships no card of its own, and a
        // client that sends nothing should get what the server would give
        // any other model.
        defaults = GenerationDefaults.house
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
        let prompt = tokenizer.encode(rendered, addBOS: false).map(Int.init)
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

        var stopMatcher = StreamingStopMatcher(stops: configuration.stopStrings)
        var content = ""
        var produced = 0
        var reason = "length"
        while produced < budget {
            let next = sampler.pick(logits, using: generator)
            if next == Int(tokenizer.eosID) { reason = "stop"; break }
            produced += 1
            let piece = tokenizer.decode([Int32(next)], skipSpecialTokens: true)
            if !piece.isEmpty {
                let visible = stopMatcher.push(piece)
                if !visible.isEmpty {
                    content += visible
                    onEvent(.content(visible))
                }
                if stopMatcher.isStopped { reason = "stop"; break }
            }
            if produced >= budget { break }
            logits = try model.step(token: next)
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        return ServerCompletion(
            content: content,
            toolCalls: [],
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: prompt.count,
                               completionTokens: produced,
                               totalTokens: prompt.count + produced,
                               cachedTokens: 0))
    }
}
