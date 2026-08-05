import Foundation
import Metal

/// Memory contract for native Qwen3.6 multi-token prediction. The target
/// model remains SSD-streamed; this plan accounts only for incremental MTP
/// state that can become resident during a request.
public struct StreamingMTPMemoryPlan: Sendable, Equatable {
    public static let allowedBudgetMiB = 256...512
    public static let defaultBudgetMiB = 384
    public static let defaultDraftKVTokens = 65_536
    public static let expertSlots = 8

    public let budgetBytes: Int
    public let residentTensorBytes: Int
    public let streamedExpertCacheBytes: Int
    public let draftKVBytes: Int
    public let targetRollbackBytes: Int
    public let scratchBytes: Int

    public var requiredBytes: Int {
        residentTensorBytes + streamedExpertCacheBytes + draftKVBytes
            + targetRollbackBytes + scratchBytes
    }
    public var headroomBytes: Int { budgetBytes - requiredBytes }

    public init(budgetMiB: Int = defaultBudgetMiB,
                residentTensorBytes: Int,
                expertStrideBytes: Int,
                draftKVTokens: Int = defaultDraftKVTokens,
                targetRollbackBytes: Int,
                scratchBytes: Int) throws {
        guard Self.allowedBudgetMiB.contains(budgetMiB) else {
            throw StreamingMTPError.invalidMemoryBudgetMiB(budgetMiB)
        }
        guard residentTensorBytes >= 0, expertStrideBytes >= 0,
              draftKVTokens > 0, targetRollbackBytes >= 0,
              scratchBytes >= 0 else {
            throw StreamingMTPError.invalidMemoryComponent
        }
        let budgetBytes = budgetMiB * 1_048_576
        let streamedExpertCacheBytes = expertStrideBytes * Self.expertSlots
        // One MTP attention layer, two FP16 KV tensors, 2 heads x 256 dims.
        let draftKVBytes = draftKVTokens * 2 * 2 * 256 * 2
        let required = residentTensorBytes + streamedExpertCacheBytes
            + draftKVBytes + targetRollbackBytes + scratchBytes
        guard required <= budgetBytes else {
            throw StreamingMTPError.memoryBudgetExceeded(requiredBytes: required,
                                                         budgetBytes: budgetBytes)
        }
        self.budgetBytes = budgetBytes
        self.residentTensorBytes = residentTensorBytes
        self.streamedExpertCacheBytes = streamedExpertCacheBytes
        self.draftKVBytes = draftKVBytes
        self.targetRollbackBytes = targetRollbackBytes
        self.scratchBytes = scratchBytes
    }
}

public enum StreamingMTPError: Error, Equatable, CustomStringConvertible {
    case invalidMemoryBudgetMiB(Int)
    case invalidMemoryComponent
    case memoryBudgetExceeded(requiredBytes: Int, budgetBytes: Int)
    case targetMustBeQwen36
    case sidecarMustBeQwen36MTP
    case greedyOnly
    case draftNotReady

    public var description: String {
        switch self {
        case .invalidMemoryBudgetMiB(let value):
            "MTP memory budget must be 256...512 MiB, got \(value)"
        case .invalidMemoryComponent:
            "MTP memory-plan components must be non-negative"
        case .memoryBudgetExceeded(let required, let budget):
            "MTP requires \(required) bytes, exceeding its \(budget)-byte budget"
        case .targetMustBeQwen36:
            "MTP target must be Qwen3.6 35B-A3B"
        case .sidecarMustBeQwen36MTP:
            "MTP sidecar has the wrong architecture"
        case .greedyOnly:
            "native MTP currently preserves exact output only for greedy decoding"
        case .draftNotReady:
            "MTP draft state has not been aligned with the target prompt"
        }
    }
}

/// Lightweight target checkpoint: target KV is append-only and only its
/// cursor is rewound; the fixed-size Gated-DeltaNet state is copied because it
/// is updated in place by the two-token verification batch.
struct SpeculativeInferenceCheckpoint: Sendable {
    let position: Int
}

struct TargetPairVerification: Sendable {
    let predictionAfterFirst: Int32
    let predictionAfterSecond: Int32
    /// Two contiguous FP16 pre-final-norm target hidden rows.
    let hiddenRows: Data
}

struct MTPPrefillResult: Sendable {
    let target: PrefillResult
    let lastTargetHidden: Data
}

public struct MTPStatistics: Sendable, Equatable {
    public private(set) var draftedTokens = 0
    public private(set) var acceptedTokens = 0
    public private(set) var targetBackbonePasses = 0
    public private(set) var emittedTokens = 0

    public var acceptanceRate: Double {
        draftedTokens == 0 ? 0 : Double(acceptedTokens) / Double(draftedTokens)
    }
    public var emittedTokensPerTargetPass: Double {
        targetBackbonePasses == 0 ? 0
            : Double(emittedTokens) / Double(targetBackbonePasses)
    }

    mutating func record(accepted: Bool, emitted: Int, targetPasses: Int) {
        draftedTokens += 1
        acceptedTokens += accepted ? 1 : 0
        targetBackbonePasses += targetPasses
        emittedTokens += emitted
    }
}

public struct MTPDecodeBatch: Sendable, Equatable {
    public let tokenIDs: [Int32]
    public let backedPrefixCount: Int
    public let acceptedDraft: Bool
    public let statistics: MTPStatistics
}

/// Target-verified greedy native-MTP session. The target always verifies the draft; a
/// rejected draft is rolled back to the GPU checkpoint captured immediately
/// after the confirmed boundary row.
public final class StreamingMTPDecoder: LogitProducer, ContextWindowReporting,
    @unchecked Sendable {
    public let target: RealForwardRunner
    let draft: RealForwardRunner
    public let memoryPlan: StreamingMTPMemoryPlan
    public private(set) var statistics = MTPStatistics()
    public let maxContext: Int
    public let draftMaxContext: Int
    private let targetConfig: ArchConfig
    private var boundaryHidden: Data?

    public init(targetModel: Model,
                mtpSidecar: Model,
                context: MetalContext,
                maxContext: Int,
                memoryBudgetMiB: Int = StreamingMTPMemoryPlan.defaultBudgetMiB,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        guard targetModel.config.family == .qwen36 else {
            throw StreamingMTPError.targetMustBeQwen36
        }
        guard mtpSidecar.config.family == .qwen36MTP else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        // Validate MTP tensors exist before attempting weight sharing
        _ = try? mtpSidecar.mtpProjection()
        _ = try? mtpSidecar.mtpEmbeddingNorm()
        _ = try? mtpSidecar.mtpHiddenNorm()
        let boundDraft = try mtpSidecar.sharingTargetWeights(from: targetModel)
        let targetRunner = try RealForwardRunner(model: targetModel,
                                                 context: context,
                                                 maxContext: maxContext,
                                                 runtimeConfiguration: runtimeConfiguration,
                                                 enableSpeculativeGDN: true)
        let draftContext = min(maxContext,
                               StreamingMTPMemoryPlan.defaultDraftKVTokens)
        let draftRuntime = RuntimeConfiguration(
            expertCacheSlots: StreamingMTPMemoryPlan.expertSlots,
            expertCachePolicy: runtimeConfiguration.expertCachePolicy,
            rdadvisePolicy: runtimeConfiguration.rdadvisePolicy,
            prefillEnabled: true,
            prefillChunkTokens: 32,
            prefillAttentionPath: runtimeConfiguration.prefillAttentionPath,
            forceLogitsHead: boundDraft.lmHeadWeightBits != 4)
        let draftRunner = try RealForwardRunner(model: boundDraft,
                                                context: context,
                                                maxContext: draftContext,
                                                runtimeConfiguration: draftRuntime)
        let scratch = PrefillChunkScratchLayout(
            config: ArchConfig.qwen36MTP,
            chunkTokens: 32).totalPersistentBytes
            + 2 * ArchConfig.qwen36MTP.vocabSize * MemoryLayout<Float16>.stride
            + 32 * ArchConfig.qwen36MTP.hiddenSize * 7 * MemoryLayout<Float16>.stride
        self.memoryPlan = try StreamingMTPMemoryPlan(
            budgetMiB: memoryBudgetMiB,
            residentTensorBytes: mtpSidecar.mtpResidentTensorBytes,
            expertStrideBytes: mtpSidecar.mtpExpertStrideBytes,
            draftKVTokens: draftContext,
            targetRollbackBytes: targetRunner.speculativeRollbackBytes,
            scratchBytes: scratch)
        self.target = targetRunner
        self.draft = draftRunner
        self.maxContext = maxContext
        self.draftMaxContext = draftContext
        self.targetConfig = targetModel.config
    }

    public func reset() {
        target.reset()
        draft.reset()
        boundaryHidden = nil
        statistics = MTPStatistics()
    }

    /// Required only for protocol compatibility. Callers should use
    /// `prepare`/`advance`; silently taking the scalar path would make an MTP
    /// session's state ambiguous.
    public func produce(token: Int32, position: Int,
                        into logits: MTLBuffer) async throws {
        throw StreamingMTPError.draftNotReady
    }

    func prepare(promptIds: [Int32],
                 config: GenerationConfig,
                 prefillConfig: PrefillRuntimeConfig,
                 logits: MTLBuffer,
                 onProgress: (Int) -> Void) async throws -> Int32 {
        guard config.isPureGreedy else { throw StreamingMTPError.greedyOnly }
        guard promptIds.count + config.maxNewTokens <= maxContext,
              promptIds.count + config.maxNewTokens <= draft.maxContext else {
            throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                                 maxNew: config.maxNewTokens,
                                                 maxContext: min(maxContext, draft.maxContext))
        }
        reset()
        let result = try await target.prefillChunkedWithMTP(
            tokens: promptIds[...],
            config: prefillConfig,
            into: logits,
            mtp: draft,
            onProgress: onProgress)
        boundaryHidden = result.lastTargetHidden
        switch result.target.seed {
        case .greedyToken(let token):
            return Int32(bitPattern: token)
        case .logitsWritten:
            return Self.argmax(logits, count: targetConfig.vocabSize)
        }
    }

    func advance(boundaryToken: Int32) async throws -> MTPDecodeBatch {
        guard let boundaryHidden else { throw StreamingMTPError.draftNotReady }
        let draftToken = try await draft.advanceMTP(
            tokens: [boundaryToken][...],
            targetHiddenRows: boundaryHidden,
            startPosition: draft.continuationPosition,
            predictNext: true)
        guard let draftToken else { throw StreamingMTPError.draftNotReady }

        let checkpoint = try target.captureSpeculativeCheckpoint(
            maximumBytes: memoryPlan.targetRollbackBytes)
        let verification = try await target.verifyGreedyPair(
            [boundaryToken, draftToken],
            startPosition: checkpoint.position)
        let rowBytes = targetConfig.hiddenSize * MemoryLayout<Float16>.stride
        let hiddenAfterBoundary = verification.hiddenRows.subdata(in: 0..<rowBytes)
        let accepted = verification.predictionAfterFirst == draftToken
        if accepted {
            _ = try await draft.advanceMTP(
                tokens: [draftToken][...],
                targetHiddenRows: hiddenAfterBoundary,
                startPosition: draft.continuationPosition,
                predictNext: false)
            self.boundaryHidden = verification.hiddenRows.subdata(in: rowBytes..<(2 * rowBytes))
            statistics.record(accepted: true, emitted: 2, targetPasses: 1)
            return MTPDecodeBatch(
                tokenIDs: [draftToken, verification.predictionAfterSecond],
                backedPrefixCount: 1,
                acceptedDraft: true,
                statistics: statistics)
        }

        // The proposal pass appended one draft KV row. It represented a
        // prediction that the target rejected, so align with Qwen's reference
        // loop by trimming it before the verified replacement is processed.
        try draft.rewindMTP(to: draft.continuationPosition - 1)
        try target.rollbackSpeculativeCheckpoint(checkpoint)
        self.boundaryHidden = hiddenAfterBoundary
        statistics.record(accepted: false, emitted: 1, targetPasses: 1)
        return MTPDecodeBatch(tokenIDs: [verification.predictionAfterFirst],
                              backedPrefixCount: 0,
                              acceptedDraft: false,
                              statistics: statistics)
    }

    private static func argmax(_ logits: MTLBuffer, count: Int) -> Int32 {
        let values = logits.contents().assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestValue = Float(values[0])
        for index in 1..<count {
            let value = Float(values[index])
            if value > bestValue {
                best = index
                bestValue = value
            }
        }
        return Int32(best)
    }

    var targetPosition: Int { target.continuationPosition }
}
