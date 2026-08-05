import Testing
@testable import NVMAI

@Suite struct StreamingMTPTests {
    @Test func defaultPlanFitsStrictBudget() throws {
        let plan = try StreamingMTPMemoryPlan(
            residentTensorBytes: 43 * 1_048_576,
            expertStrideBytes: 1_769_472,
            targetRollbackBytes: 62 * 1_048_576,
            scratchBytes: 8 * 1_048_576)
        #expect(plan.streamedExpertCacheBytes == 8 * 1_769_472)
        #expect(plan.draftKVBytes == 128 * 1_048_576)
        #expect(plan.requiredBytes < plan.budgetBytes)
        #expect(plan.headroomBytes > 0)
    }

    @Test func oversizedResidentDesignIsRejected() {
        #expect(throws: StreamingMTPError.self) {
            _ = try StreamingMTPMemoryPlan(
                budgetMiB: 256,
                residentTensorBytes: 300 * 1_048_576,
                expertStrideBytes: 1_769_472,
                targetRollbackBytes: 62 * 1_048_576,
                scratchBytes: 8 * 1_048_576)
        }
    }

    @Test func architectureIsOneLayerAndBounded() {
        let arch = ArchConfig.qwen36MTP
        #expect(arch.family == .qwen36MTP)
        #expect(arch.numLayers == 1)
        #expect(arch.fullAttentionLayerMask == [1])
        #expect(arch.slidingWindow == 65_536)
        #expect(arch.topKExperts == 8)
        #expect(!arch.tieWordEmbeddings)
    }

    @Test func statisticsMeasureAcceptedSpeedup() {
        var statistics = MTPStatistics()
        statistics.record(accepted: true, emitted: 2, targetPasses: 1)
        statistics.record(accepted: false, emitted: 1, targetPasses: 1)
        #expect(statistics.acceptanceRate == 0.5)
        #expect(statistics.emittedTokensPerTargetPass == 1.5)
    }
}
