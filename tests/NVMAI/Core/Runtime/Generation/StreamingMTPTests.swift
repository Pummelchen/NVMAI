import Testing
import Foundation
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
        // The default budget (384 MiB) must cover the sum of every component.
        #expect(plan.requiredBytes == 43 * 1_048_576 + 8 * 1_769_472
            + 128 * 1_048_576 + 62 * 1_048_576 + 8 * 1_048_576)
        #expect(plan.requiredBytes < plan.budgetBytes)
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

    // MARK: - Toy sidecar integration (T27)

    /// The MTP sidecar toy mirrors the real installed sidecar schema: it
    /// loads under the qwen36_mtp expectation, resolves its adapter tensors,
    /// and shares the target's embedding/lm_head buffers (weight sharing, no
    /// copy) via `sharingTargetWeights`.
    @Test func mtpSidecarLoadsAndSharesTargetWeights() throws {
        let targetDir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: targetDir) }
        let sidecarDir = try QwenToySynthetic.writeMTP()
        defer { try? FileManager.default.removeItem(at: sidecarDir) }
        let ctx = try MetalContext()
        let target = try Model.load(directoryURL: targetDir,
                                    device: ctx.device,
                                    expecting: .qwenToy())
        let sidecar = try Model.load(directoryURL: sidecarDir,
                                     device: ctx.device,
                                     expecting: .qwenToyMTP())

        #expect(target.config.family == .qwen36)
        #expect(sidecar.config.family == .qwen36MTP)
        // MTP adapter tensors resolve against the sidecar's resident index.
        _ = try sidecar.mtpProjection()
        _ = try sidecar.mtpEmbeddingNorm()
        _ = try sidecar.mtpHiddenNorm()

        let bound = try sidecar.sharingTargetWeights(from: target)
        #expect(try bound.embedding().buffer === target.embedding().buffer)
        #expect(try bound.lmHead().buffer === target.lmHead().buffer)
    }

    /// Full MTP decode-loop smoke on the toy pair: load the target and the
    /// sidecar, construct `StreamingMTPDecoder`, and run a short greedy
    /// completion via `runRawCompletion` (which routes StreamingMTPDecoder
    /// producers through `runStreamingMTPCompletion`). Asserts tokens are
    /// emitted and the acceptance metrics are reported on the decoder.
    @Test func mtpDecodeLoopEmitsTokensAndReportsAcceptance() async throws {
        let targetDir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: targetDir) }
        let sidecarDir = try QwenToySynthetic.writeMTP()
        defer { try? FileManager.default.removeItem(at: sidecarDir) }
        let ctx = try MetalContext()
        let tokenizer = try await GFTokenizer.load(
            from: ChatMLTemplateTests.fixtureFolder())
        let target = try Model.load(directoryURL: targetDir,
                                    device: ctx.device,
                                    expecting: .qwenToy())
        let sidecar = try Model.load(directoryURL: sidecarDir,
                                     device: ctx.device,
                                     expecting: .qwenToyMTP())
        let decoder = try StreamingMTPDecoder(
            targetModel: target,
            mtpSidecar: sidecar,
            context: ctx,
            maxContext: 64)
        let scratch = try RawCompletionScratch(context: ctx,
                                               vocab: tokenizer.vocabSize)

        var emitted = 0
        let result = try await runRawCompletion(
            producer: decoder,
            tokenizer: tokenizer,
            promptIds: [1, 2],
            config: GenerationConfig(maxNewTokens: 4, temperature: 0),
            context: ctx,
            scratch: scratch,
            onProgress: { progress in
                if case .token = progress { emitted += 1 }
            })

        #expect(result.newTokens > 0, "MTP loop emitted no tokens")
        #expect(emitted > 0)
        // Greedy verification always runs at least one advance for a 4-token
        // budget, so the acceptance statistics are reported (drafted >= 1).
        #expect(decoder.statistics.draftedTokens > 0,
                "no draft/verify cycle ran; acceptance metrics were not reported")
        #expect(decoder.statistics.acceptedTokens >= 0)
        #expect(decoder.statistics.emittedTokens >= result.newTokens)
        #expect(decoder.targetPosition == decoder.target.continuationPosition)
    }
}
