import Testing
@testable import NVMAI

/// The slot default is derived from a RAM budget and the model's own expert
/// stride, so one rule serves every quantisation.
///
/// The budget is 8 GiB because expert reads bypass the page cache, leaving the slot
/// cache as the only cache: a routing trace measured 131 distinct experts per layer
/// over a 128-token window, and 128 slots is the first budget that holds it.
/// Measured bounded, 4-bit, short prompt: 8.73 tok/s at 16 slots against 18.91 at
/// 128. Under the page-cache policy the ordering inverts (13.61 at 16 against 8.78
/// at 128), so these numbers are only right while reads bypass the cache.
@Suite struct ExpertCacheBudgetTests {
    static let layers = 40
    static let stride4 = UInt64(1_769_472)
    static let stride6 = UInt64(2_555_904)
    static let stride8 = UInt64(3_342_336)

    @Test func defaultBudgetLandsOnTheMeasuredOptimumPerQuant() {
        // 4-bit holds the working set at 128 slots (8.44 GiB).
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride4, layers: Self.layers) == 128)
        // 8-bit cannot: 128 slots would be 15.94 GiB and measured 1.22 tok/s
        // thrashing on a 24 GB machine, so the budget caps it at 64.
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride8, layers: Self.layers) == 64)
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
    /// 8-bit at the default budget must stay inside RAM. 128 slots is 15.94 GiB,
    /// which measured 1.22 tok/s against 5.21 at 64 -- a cliff, not a slope.
    @Test func eightBitDefaultStaysBelowTheThrashingPoint() {
        let slots = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride8, layers: Self.layers)
        let bytes = Double(slots) * Double(Self.stride8) * Double(Self.layers)
        #expect(bytes / 1_073_741_824 < 12.0,
                "8-bit default would reserve \(bytes / 1_073_741_824) GiB")
    }

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
