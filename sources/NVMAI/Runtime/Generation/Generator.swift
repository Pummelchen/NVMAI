import Foundation

public enum StopReason: String, Codable, Sendable, Equatable {
    case eos
    case endOfTurn
    case maxTokens
    case stopString
    case toolCalls
    /// The caller's `shouldStop()` closure returned true (external stop
    /// signal) before any configured stop string matched.
    case external
}

enum GeneratorError: Error, CustomStringConvertible, Equatable {
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    case invalidGenerationConfig(String)
    case invalidContinuation(String)
    case emptyPrompt
    case invalidSamplerPath(String)
    /// Every logit in the sampled row was non-finite, so the softmax produced
    /// no probability mass and the sampler could only return its in-range
    /// fallback. Reported instead of emitting that fallback as a token: a
    /// generation stuck on token 0 is indistinguishable from a bad prompt or a
    /// bad temperature, and the underlying cause is not the sampler.
    case degenerateLogitsRow
    /// The sampler answered with an id outside the vocabulary.
    ///
    /// Its kernels are not supposed to be able to: they clamp their sentinel to
    /// token 0 and the tests pin "whatever the logits, the sampler returns an
    /// index inside the vocabulary". An out-of-range id has nevertheless been
    /// observed intermittently under full-suite GPU load and never identified.
    /// This is the consumer's side of that contract -- the id indexes the
    /// vocabulary and extends the KV history, so using one is silent corruption.
    /// Reported with the id and the vocabulary size, whatever produced it, which
    /// is what turns an unidentified intermittent fault into a named one.
    case samplerReturnedOutOfRangeToken(id: UInt32, vocab: Int)

    public var description: String {
        switch self {
        case .contextOverflow(let prompt, let maxNew, let maxContext):
            return "context overflow: prompt \(prompt) + maxNew \(maxNew) exceeds maxContext \(maxContext)"
        case .invalidGenerationConfig(let reason):
            return reason
        case .invalidContinuation(let reason):
            return reason
        case .emptyPrompt:
            return "empty prompt"
        case .invalidSamplerPath(let value):
            return "unsupported sampler path '\(value)'; allowed: tiled, generic"
        case .samplerReturnedOutOfRangeToken(let id, let vocab):
            return "sampler returned token id \(id) outside the vocabulary (\(vocab) entries); "
                + "the id would index past the embedding and the KV history"
        case .degenerateLogitsRow:
            return "sampler row had no finite logit: every value in the row was "
                + "NaN (or +inf with logit softcap disabled), so no token could "
                + "be drawn. The model produced a degenerate distribution; "
                + "check the install for corrupt or NaN weights."
        }
    }
}
