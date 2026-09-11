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

/// `--ram-budget` is the user-facing knob, so its parser has to accept what people
/// type and reject what they mistype -- a silently misparsed size would produce a
/// tiny cache and a 2.2x throughput loss with no error.
@Suite struct BudgetParsingTests {
    @Test func acceptsTheUnitsUsersType() {
        #expect(RuntimeConfiguration.parseBudgetBytes("8G") == 8 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("2g") == 2 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("512M") == 512 << 20)
        #expect(RuntimeConfiguration.parseBudgetBytes("8GiB") == 8 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("1024KB") == 1024 << 10)
        #expect(RuntimeConfiguration.parseBudgetBytes(" 4G ") == 4 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("1073741824") == 1 << 30)
    }

    @Test func acceptsFractionalSizes() {
        #expect(RuntimeConfiguration.parseBudgetBytes("1.5G") == Int(1.5 * Double(1 << 30)))
        #expect(RuntimeConfiguration.parseBudgetBytes("0.5G") == 1 << 29)
    }

    @Test func rejectsWhatWouldSilentlyShrinkTheCache() {
        for bad in ["", " ", "bogus", "G", "-2G", "0", "0G", "abcG", "2X", "2GG"] {
            #expect(RuntimeConfiguration.parseBudgetBytes(bad) == nil,
                    "\(bad.debugDescription) should not parse")
        }
    }

    /// The knob has to move the outcome, and monotonically.
    @Test func budgetDrivesTheSlotCount() {
        let stride = UInt64(1_769_472)
        let one = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: stride, layers: 40, budgetBytes: 1 << 30)
        let four = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: stride, layers: 40, budgetBytes: 4 << 30)
        let eight = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: stride, layers: 40, budgetBytes: 8 << 30)
        #expect(one == 16)
        #expect(four == 64)
        #expect(eight == 128)
        #expect(one < four && four < eight)
    }
}

/// The per-family decode tuning, and the clamp that keeps it honest on a
/// machine smaller than the one it was measured on.
///
/// Measured 2026-08-30, 512-token prose, fresh process per run, interleaved:
/// qwen38flash 4-bit 5.735 -> 6.957 tok/s with 12 GiB + depth 1; qwen36 8-bit
/// 11.994 -> 12.659 with depth 1; qwen36 4-bit regressed 23.234 -> 22.248, so
/// it ships without prefetch.
@Suite struct DecodeTuningTests {
    static let qwen38Stride = UInt64(2_768_896)
    static let qwen38Layers = 48

    @Test func qwen38FlashGetsTheWiderCacheAndPrefetch() {
        let tuning = RuntimeConfiguration.decodeTuning(family: .qwen38flash,
                                                       weightBits: 4)
        #expect(tuning.expertCacheBudgetBytes == 12 << 30)
        #expect(tuning.prefetchDepth == 1)
        // 12 GiB has to actually land on 96 slots for this payload, which is
        // the whole point of the entry.
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.qwen38Stride, layers: Self.qwen38Layers,
            budgetBytes: tuning.expertCacheBudgetBytes) == 96)
    }

    @Test func prefetchShipsOnlyWhereItMeasuredFaster() {
        // 8-bit streams twice the bytes, so expert I/O is a large enough share
        // of the token for a speculative read to pay.
        #expect(RuntimeConfiguration.decodeTuning(
            family: .qwen36, weightBits: 8).prefetchDepth == 1)
        #expect(RuntimeConfiguration.decodeTuning(
            family: .qwen36MTP, weightBits: 8).prefetchDepth == 1)
        // At 4-bit the same family spends ~7 ms of a ~44 ms token on expert
        // I/O; prefetch measured -3.9% (Ornith) and -4.2% (Qwen 3.6).
        #expect(RuntimeConfiguration.decodeTuning(
            family: .qwen36, weightBits: 4).prefetchDepth == 0)
        #expect(RuntimeConfiguration.decodeTuning(
            family: .qwen36MTP, weightBits: 4).prefetchDepth == 0)
    }

    @Test func the35BFamiliesKeepTheirEstablishedBudget() {
        for bits in [4, 8] {
            #expect(RuntimeConfiguration.decodeTuning(
                family: .qwen36, weightBits: bits).expertCacheBudgetBytes
                == RuntimeConfiguration.defaultExpertCacheBudgetBytes)
        }
    }

    @Test func budgetIsClampedToWhatTheMachineCanHold() {
        let wanted = 12 << 30
        // 24 GiB and up: the tuned budget survives intact.
        #expect(RuntimeConfiguration.affordableExpertCacheBudget(
            wanted, physicalMemory: 32 << 30) == wanted)
        // 16 GiB: half is 8 GiB, so the 12 GiB aimed at qwen38flash is cut
        // rather than handed to a machine with no room for it.
        #expect(RuntimeConfiguration.affordableExpertCacheBudget(
            wanted, physicalMemory: 16 << 30) == 8 << 30)
        // And the clamp has to change the slot count, not just the number.
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.qwen38Stride, layers: Self.qwen38Layers,
            budgetBytes: RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 16 << 30)) == 64)
    }

    @Test func clampNeverGrowsABudget() {
        #expect(RuntimeConfiguration.affordableExpertCacheBudget(
            8 << 30, physicalMemory: 128 << 30) == 8 << 30)
    }
}
