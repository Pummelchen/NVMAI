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
        }
    }
}
