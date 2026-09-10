import Foundation

/// Which reasoning controls a family's chat template truthfully defines.
/// NVMAI exposes exactly what the installed template implements: Ornith 1.5
/// and Qwen 3.6 define only the binary `enable_thinking` switch, while the
/// Qwen3.8-Flash-Next template additionally defines `reasoning_effort`
/// low|medium|xhigh (default xhigh) while thinking is on. Verified against
/// the pinned upstream `chat_template.jinja` on 2026-08-27; see
/// docs/qwen38-flash-next-port.md. Re-verified 2026-09-11 against every
/// installed template (Qwen 3.6, AgentWorld, Ornith 1.5, Qwen3.8, and the
/// CPU engine's Qwen3.5): ReasoningLevelTemplateTests renders each one.
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

    /// Levels this family's chat template actually honours, `.off` first.
    public var supportedReasoningLevels: [ReasoningLevel] {
        reasoningControl.supportedLevels
    }

    /// Runtime settings for a level; throws for a level the family does not
    /// support.
    public func runtimeReasoning(for level: ReasoningLevel) throws
        -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
        try reasoningControl.runtimeReasoning(for: level, family: rawValue)
    }
}

/// One vocabulary of thinking levels for every family, so a configurator can
/// offer a single picker. Each family answers which of these its template
/// renders differently; the others are refused rather than mapped to a
/// neighbour, because a level that does not change the prompt does not
/// change the model's behaviour either.
public enum ReasoningLevel: String, CaseIterable, Sendable, Codable {
    case off, on, minimal, low, medium, high, xhigh, max

    public var displayName: String {
        self == .xhigh ? "extra high" : rawValue
    }

    init(effort: ModelReasoningEffort) {
        // Exhaustive on purpose: a new effort case must be given its level
        // here before it can compile.
        switch effort {
        case .low: self = .low
        case .medium: self = .medium
        case .xhigh: self = .xhigh
        }
    }
}

public enum ReasoningLevelError: Error, Equatable, CustomStringConvertible {
    case unsupported(family: String, level: ReasoningLevel,
                     supported: [ReasoningLevel])

    public var description: String {
        switch self {
        case .unsupported(let family, let level, let supported):
            return "thinking level \(level.displayName) is not supported: the "
                + "\(family) chat template renders only "
                + supported.map(\.displayName).joined(separator: ", ")
        }
    }
}

extension ModelReasoningControl {
    /// A binary template renders two prompts: the closed and the open think
    /// block. An effort template renders off plus one prompt per effort it
    /// accepts. Plain `.on` is not a level there: it selects the template's
    /// default effort and renders byte-identically to naming that effort, so
    /// offering both would put two entries on one prompt.
    ///
    /// `ModelReasoningEffort` enumerates exactly what the one effort
    /// template here (Qwen3.8) accepts; it raises on any other value. A
    /// second effort family with a different set would need its own list in
    /// the control case.
    var supportedLevels: [ReasoningLevel] {
        switch self {
        case .binaryThinking:
            return [.off, .on]
        case .thinkingWithEffortLevels:
            return [.off] + ModelReasoningEffort.allCases.map(ReasoningLevel.init(effort:))
        }
    }

    /// Shared by the GPU and CPU families so the two cannot disagree on what
    /// a level means.
    func runtimeReasoning(for level: ReasoningLevel, family: String) throws
        -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
        let supported = supportedLevels
        guard supported.contains(level) else {
            throw ReasoningLevelError.unsupported(
                family: family, level: level, supported: supported)
        }
        switch level {
        case .off: return (.off, nil)
        case .on: return (.on, nil)
        default:
            // Every supported level other than off/on is an effort level by
            // construction of `supportedLevels`, and shares its raw value.
            return (.on, ModelReasoningEffort(rawValue: level.rawValue))
        }
    }
}
