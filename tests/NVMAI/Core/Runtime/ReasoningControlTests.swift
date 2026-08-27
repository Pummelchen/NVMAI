import Testing
@testable import NVMAI

/// The per-family reasoning contract: templates that define only the binary
/// thinking switch reject effort levels instead of faking them, and the
/// Qwen3.8-Flash-Next template's levels resolve with its xhigh default.
@Suite("Reasoning control")
struct ReasoningControlTests {
    @Test("Family capability mapping")
    func familyMapping() {
        #expect(ModelFamily.qwen36.reasoningControl == .binaryThinking)
        #expect(ModelFamily.qwen36MTP.reasoningControl == .binaryThinking)
        #expect(ModelFamily.qwen38flash.reasoningControl
            == .thinkingWithEffortLevels(defaultEffort: .xhigh))
    }

    @Test("Nil effort always validates")
    func nilEffortPasses() throws {
        for family in [ModelFamily.qwen36, .qwen36MTP, .qwen38flash] {
            try family.validateReasoning(thinkingMode: .off, effort: nil)
            try family.validateReasoning(thinkingMode: .on, effort: nil)
        }
    }

    @Test("Binary families reject every effort level")
    func binaryFamiliesRejectEffort() {
        for effort in ModelReasoningEffort.allCases {
            #expect(throws: ModelReasoningControlError.effortUnsupported(
                family: .qwen36, effort: effort)) {
                try ModelFamily.qwen36.validateReasoning(
                    thinkingMode: .on, effort: effort)
            }
        }
    }

    @Test("Effort families require thinking on")
    func effortRequiresThinking() throws {
        #expect(throws: ModelReasoningControlError.effortRequiresThinkingOn(
            effort: .low)) {
            try ModelFamily.qwen38flash.validateReasoning(
                thinkingMode: .off, effort: .low)
        }
        try ModelFamily.qwen38flash.validateReasoning(
            thinkingMode: .on, effort: .low)
    }

    @Test("Effective effort resolves the template default")
    func effectiveEffort() {
        #expect(ModelFamily.qwen38flash.effectiveReasoningEffort(
            thinkingMode: .on, effort: nil) == .xhigh)
        #expect(ModelFamily.qwen38flash.effectiveReasoningEffort(
            thinkingMode: .on, effort: .medium) == .medium)
        #expect(ModelFamily.qwen38flash.effectiveReasoningEffort(
            thinkingMode: .off, effort: .low) == nil)
        #expect(ModelFamily.qwen36.effectiveReasoningEffort(
            thinkingMode: .on, effort: nil) == nil)
    }

    @Test("Environment resolution accepts only the template's levels")
    func environmentResolution() {
        #expect(ModelReasoningEffort.resolved(environment: [:]) == nil)
        #expect(ModelReasoningEffort.resolved(
            environment: ["NVMAI_REASONING_EFFORT": "XHigh"]) == .xhigh)
        #expect(ModelReasoningEffort.resolved(
            environment: ["NVMAI_REASONING_EFFORT": "high"]) == nil)
    }
}
