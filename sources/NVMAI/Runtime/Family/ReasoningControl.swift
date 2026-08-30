import Foundation

/// Which reasoning controls a family's chat template truthfully defines.
/// NVMAI exposes exactly what the installed template implements: Ornith 1.5
/// and Qwen 3.6 define only the binary `enable_thinking` switch, while the
/// Qwen3.8-Flash-Next template additionally defines `reasoning_effort`
/// low|medium|xhigh (default xhigh) while thinking is on. Verified against
/// the pinned upstream `chat_template.jinja` on 2026-08-27; see
/// docs/qwen38-flash-next-port.md.
public enum ModelReasoningControl: Sendable, Equatable {
    case binaryThinking
    case thinkingWithEffortLevels(defaultEffort: ModelReasoningEffort)
}

public enum ModelReasoningControlError: Error, Equatable, CustomStringConvertible {
    case effortUnsupported(family: ModelFamily, effort: ModelReasoningEffort)
    case effortRequiresThinkingOn(effort: ModelReasoningEffort)

    public var description: String {
        switch self {
        case .effortUnsupported(let family, let effort):
            return "reasoning effort \(effort.rawValue) is not supported: the "
                + "\(family.rawValue) chat template defines only the binary "
                + "thinking off|on switch"
        case .effortRequiresThinkingOn(let effort):
            return "reasoning effort \(effort.rawValue) requires thinking on; "
                + "the chat template ignores effort while thinking is off"
        }
    }
}

extension ModelFamily {
    public var reasoningControl: ModelReasoningControl {
        switch self {
        case .qwen36, .qwen36MTP:
            return .binaryThinking
        case .qwen38flash, .qwen38flashMTP:
            return .thinkingWithEffortLevels(defaultEffort: .xhigh)
        }
    }

    /// The effort a template applies for this family under the given
    /// settings: the explicit override, else the template default, else nil
    /// for binary families or while thinking is off.
    public func effectiveReasoningEffort(
        thinkingMode: ModelThinkingMode,
        effort: ModelReasoningEffort?
    ) -> ModelReasoningEffort? {
        guard thinkingMode.isEnabled,
              case .thinkingWithEffortLevels(let defaultEffort) = reasoningControl else {
            return nil
        }
        return effort ?? defaultEffort
    }

    /// Rejects a reasoning-effort request this family's template does not
    /// define. A nil effort always passes: it means the binary control, or
    /// the template's own default level for effort-aware families.
    public func validateReasoning(thinkingMode: ModelThinkingMode,
                                  effort: ModelReasoningEffort?) throws {
        guard let effort else { return }
        switch reasoningControl {
        case .binaryThinking:
            throw ModelReasoningControlError.effortUnsupported(
                family: self, effort: effort)
        case .thinkingWithEffortLevels:
            guard thinkingMode.isEnabled else {
                throw ModelReasoningControlError.effortRequiresThinkingOn(
                    effort: effort)
            }
        }
    }
}
