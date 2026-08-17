import Testing
@testable import NVMAI

/// The slot default is derived from a RAM budget and the model's own expert
/// stride, so the same rule lands on the measured optimum for each quantisation.
/// These pin that rule, because the numbers it produces were expensive to find:
/// benchmarked at the shipped 262144 context, 4-bit reached 13.61 tok/s at 16
/// slots against 9.85 at 64, and the old fixed default of 64 was slower *and*
/// four times larger.
@Suite struct ExpertCacheBudgetTests {
    static let layers = 40
    static let stride4 = UInt64(1_769_472)
    static let stride6 = UInt64(2_555_904)
    static let stride8 = UInt64(3_342_336)

    @Test func defaultBudgetLandsOnTheMeasuredOptimumPerQuant() {
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride4, layers: Self.layers) == 16)
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride8, layers: Self.layers) == 8)
    }

    @Test func everyResultIsASupportedSlotCount() {
        for stride in [Self.stride4, Self.stride6, Self.stride8] {
            for budget in [1 << 28, 1 << 30, 1 << 32, 1 << 34] {
                let slots = RuntimeConfiguration.expertCacheSlots(
                    expertStrideBytes: stride, layers: Self.layers,
                    budgetBytes: budget)
                #expect(RuntimeConfiguration.allowedExpertCacheSlots.contains(slots),
                        "stride \(stride) budget \(budget) gave \(slots)")
            }
        }
    }

    @Test func biggerBudgetNeverShrinksTheCache() {
        var previous = 0
        for budget in [1 << 27, 1 << 28, 1 << 29, 1 << 30, 1 << 31, 1 << 32] {
            let slots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.stride4, layers: Self.layers,
                budgetBytes: budget)
            #expect(slots >= previous, "budget \(budget) went backwards")
            previous = slots
        }
    }

    /// A degenerate manifest must not produce a zero or negative slot count --
    /// that would reach the streamer as an empty cache.
    @Test func degenerateInputsFallBackToTheSmallestSupportedCount() {
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: 0, layers: Self.layers) == 8)
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride4, layers: 0) == 8)
    }

    /// The budget must be honoured within one step of the allowed ladder, or the
    /// footprint promise is empty.
    @Test func chosenCountStaysNearTheRequestedBudget() {
        let budget = 1 << 30
        for stride in [Self.stride4, Self.stride8] {
            let slots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: stride, layers: Self.layers, budgetBytes: budget)
            let actual = Double(slots) * Double(stride) * Double(Self.layers)
            #expect(actual <= Double(budget) * 1.15,
                    "stride \(stride): \(actual / 1_073_741_824) GiB exceeds budget")
        }
    }
}
