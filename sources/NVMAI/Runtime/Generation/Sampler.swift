import Foundation
import Metal
import Synchronization

/// Canonical sampling defaults shared by every NVMAI model and quantization.
public enum GenerationDefaults {
    public static let temperature: Float = 0.6
    public static let topK = 20
    public static let topP: Float = 0.95
    public static let presencePenalty: Float = 0
}

/// Generation knobs threaded from the caller through the `Generator` into the
/// sampler. Pure value type; one per `generate(...)` call.
///
/// Canonical home is here (the sampler is the primary consumer); `Generator`
/// reuses the same type rather than redeclaring it.
public struct GenerationConfig: Sendable {
    public var maxNewTokens: Int = 256
    public var temperature: Float = GenerationDefaults.temperature
    public var topK: Int? = GenerationDefaults.topK
    public var topP: Float? = GenerationDefaults.topP
    /// OpenAI-compatible presence penalty. NVMAI currently supports the
    /// neutral value only; keeping it explicit prevents front ends from
    /// silently drifting to a different policy.
    public var presencePenalty: Float = GenerationDefaults.presencePenalty
    public var repetitionPenalty: Float = 1.0
    public var seed: UInt64? = nil         // nil = nondeterministic
    public var stopStrings: [String] = []
    public var extraStopTokens: Set<Int32> = []

    public init(maxNewTokens: Int = 256,
                temperature: Float = GenerationDefaults.temperature,
                topK: Int? = GenerationDefaults.topK,
                topP: Float? = GenerationDefaults.topP,
                presencePenalty: Float = GenerationDefaults.presencePenalty,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stopStrings: [String] = [],
                extraStopTokens: Set<Int32> = []) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stopStrings = stopStrings
        self.extraStopTokens = extraStopTokens
    }

    public func validate() throws {
        guard maxNewTokens > 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "maxNewTokens must be greater than zero")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "temperature must be finite and nonnegative")
        }
        if let topK, !(1...256).contains(topK) {
            throw GeneratorError.invalidGenerationConfig(
                "topK must be between 1 and 256")
        }
        if let topP, (!topP.isFinite || topP <= 0 || topP > 1) {
            throw GeneratorError.invalidGenerationConfig(
                "topP must be greater than zero and at most one")
        }
        guard presencePenalty.isFinite, presencePenalty == 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "presencePenalty must be zero; nonzero presence penalties are not implemented")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw GeneratorError.invalidGenerationConfig(
                "topP below one requires topK; full-vocabulary nucleus sampling is not implemented")
        }
    }

}

/// Which path a `sample(...)` call took.
enum SamplePath: Sendable, Equatable {
    case greedyGPU
    case gpuSampled
    case hostPenalty
}

/// Which Top-K implementation serves a `1...64` sampled request.
///
/// `tiled` is production: a three-stage reduction that keeps the top 64 of
/// every 1,024-entry tile, so the whole vocabulary reaches one final tile in
/// three dispatches. `generic` forces the older single-threadgroup kernel that
/// extracts Top-K in k full vocabulary passes.
///
/// The two are required to agree token-for-token — `SampleTopK64Tests` pins
/// that across k, temperature, and seed — so this exists to measure the
/// difference, not to choose behavior. It is the control arm that made the
/// +30.07% (4-bit) / +11.36% (8-bit) qualification an interleaved same-binary
/// A/B instead of a comparison against a separately-built baseline.
public enum RuntimeSamplerPath: String, Codable, Sendable {
    case tiled
    case generic

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeSamplerPath {
        guard let raw = environment["NVMAI_SAMPLER_PATH"] else { return .tiled }
        guard let value = RuntimeSamplerPath(rawValue: raw) else {
            throw GeneratorError.invalidSamplerPath(raw)
        }
        return value
    }
}

/// Turns `GenerationConfig` + a logits buffer into one token id, staying
/// GPU-resident wherever the kernels allow.
///
/// The built `sample` kernel already does temperature / top-k / top-p / seeded
/// draw / greedy argmax on GPU reading softmaxed probs, so this type's job is:
/// (1) run the softcap+softmax front-end (`logit_softcap_softmax`), (2) apply
/// repetition penalty — the one policy that needs `history` random access — as
/// a single in-place CPU pass over the (shared) logits before the front-end,
/// and (3) derive a per-position seed so a fixed `seed` is reproducible across
/// token positions.
///
/// The chosen id lands in a 1-element UInt32 buffer. The generation loop reads
/// that value after the command buffer completes.
///
/// Truncation follows mlx-lm's sampler order: Top-P is computed from the
/// model's full probability distribution, Top-K caps that surviving set, and
/// temperature is applied only to the final categorical draw.
final class Sampler {
    private let softcap: LogitSoftcapSoftmax
    private let softcapTiled: LogitSoftcapSoftmaxTiled?
    private let sampleKernel: Sample
    private let topK64Kernel: SampleTopK64
    private let samplerPath: RuntimeSamplerPath
    let vocab: Int
    private let logitSoftcap: Float

    /// Incremental repetition-penalty history (R25): id -> occurrence count,
    /// carried across `sample` calls within one generation. The penalty is
    /// applied once per distinct id (HF convention), so the counts themselves
    /// are bookkeeping; the dict replaces the per-token `Set(history)` build.
    private var penaltyFrequency: [Int32: Int] = [:]
    /// Whether the current generation's prompt has been folded into
    /// `penaltyFrequency` yet.
    private var penaltyHistorySeeded = false

    /// Monotonic counter combined with the clock so two samples in the same
    /// nanosecond still draw distinct non-deterministic seeds (R34).
    private static let nondeterministicSeedCounter = Atomic<UInt64>(0)

    init(context: MetalContext, vocab: Int = 262_144,
                logitSoftcap: Float = 30.0) throws {
        self.softcap = try LogitSoftcapSoftmax(context: context)
        self.softcapTiled = try LogitSoftcapSoftmaxTiled(context: context, vocab: vocab)
        self.sampleKernel = try Sample(context: context)
        self.topK64Kernel = try SampleTopK64(context: context, vocab: vocab)
        self.samplerPath = try RuntimeSamplerPath.environmentValue()
        self.vocab = vocab
        self.logitSoftcap = logitSoftcap
    }

    /// Encode the sampler onto `commandBuffer`. `logits` is FP16 [vocab],
    /// post-lm_head and pre-softcap, in a `.storageModeShared` buffer (the
    /// repetition-penalty path edits it in place). `probs` is a preallocated
    /// FP16 [vocab] scratch. `outToken` holds one UInt32. `position` indexes the
    /// per-position seed advance. Returns the path taken.
    @discardableResult
    func sample(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       probs: MTLBuffer,
                       history: [Int32],
                       config: GenerationConfig,
                       position: Int,
                       outToken: MTLBuffer) throws -> SamplePath {
        let v = UInt32(vocab)

        let appliedPenalty = config.repetitionPenalty != 1.0 && !history.isEmpty
        if appliedPenalty {
            applyRepetitionPenaltyInPlace(logits: logits,
                                          history: history,
                                          penalty: config.repetitionPenalty)
        }
        // The tiled front-end follows the same path selection as the Top-K
        // half: `generic` forces the single-threadgroup pair so an A/B
        // measures both halves of the sampler, not one.
        if samplerPath == .tiled, let softcapTiled {
            try softcapTiled.encode(commandBuffer: commandBuffer,
                                    logits: logits, probs: probs, v: v,
                                    softcap: logitSoftcap)
        } else {
            try softcap.encode(commandBuffer: commandBuffer,
                               logits: logits, probs: probs, v: v,
                               softcap: logitSoftcap)
        }

        let isGreedy = config.temperature == 0
        let seed = Self.seedFor(config: config, position: position)
        // The tiled reduction serves every k it can reconstruct from a
        // top-64-per-tile stage 1, which is all of 1...64 — not just 64. The
        // production default is Top-K 20, so gating on `== 64` sent every
        // shipped token to the generic kernel instead, and that kernel takes
        // k full passes over a 262,144-entry vocabulary from a single
        // 256-thread threadgroup. Measured at the published 4-bit profile,
        // widening this gate is worth 16.656 -> 21.454 tok/s (+28.8%), with
        // the head_logits->embed gap falling from 15.45 ms to 1.41 ms/token.
        // k > 64, k == 0 (top-k disabled, k becomes 256), and greedy stay on
        // the generic path, which remains the reference implementation.
        if samplerPath == .tiled,
           config.temperature > 0,
           let requestedK = config.topK,
           (1...64).contains(requestedK) {
            try topK64Kernel.encode(commandBuffer: commandBuffer,
                                    probs: probs,
                                    outToken: outToken,
                                    temperature: config.temperature,
                                    topP: config.topP ?? 1.0,
                                    seed: seed,
                                    topK: UInt32(requestedK))
        } else {
            try sampleKernel.encode(commandBuffer: commandBuffer,
                                    probs: probs, outToken: outToken, v: v,
                                    temperature: isGreedy ? 0.0 : config.temperature,
                                    topK: UInt32(config.topK ?? 0),
                                    topP: config.topP ?? 1.0,
                                    seed: seed,
                                    position: UInt32(position))
        }

        if appliedPenalty { return .hostPenalty }
        return isGreedy ? .greedyGPU : .gpuSampled
    }

    // MARK: - Repetition penalty (host, in place)

    /// HF convention: for each token id seen in `history`, a positive logit is
    /// divided by `penalty`, a negative logit multiplied. Edits the shared
    /// `logits` buffer in place — no full-buffer copy, only the unique history
    /// entries are touched.
    ///
    /// The penalty must act on the POST-softcap logit (HF applies it to the
    /// model's output logits). For architectures with a logit softcap the raw
    /// logits can reach deep into tanh saturation, where dividing the raw value
    /// by 1.1 moves the capped logit by ~nothing — the penalty silently no-ops
    /// on exactly the high-confidence tokens that form repetition loops. So:
    /// softcap the raw value, penalize, and invert through atanh so the
    /// downstream softcap+softmax kernel reproduces the penalized capped logit.
    /// (Qwen 3.6 has no logit softcap, so the 0.0 branch is the production
    /// path.)
    ///
    /// Incremental history (R25): the first call of a generation folds the
    /// prompt into `penaltyFrequency`; each later call only adds the single
    /// token appended since the last call (the history's last element). The
    /// logit edit still runs for every distinct id every call because the
    /// logits buffer is fresh per token.
    private func applyRepetitionPenaltyInPlace(logits: MTLBuffer,
                                               history: [Int32],
                                               penalty: Float) {
        if !penaltyHistorySeeded {
            for id in history where id >= 0 && Int(id) < vocab {
                penaltyFrequency[id, default: 0] += 1
            }
            penaltyHistorySeeded = true
        } else if let last = history.last, last >= 0, Int(last) < vocab {
            penaltyFrequency[last, default: 0] += 1
        }

        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        for (id, _) in penaltyFrequency {
            guard id >= 0 && Int(id) < vocab else { continue }
            let i = Int(id)
            let z = Float(ptr[i])
            let penalized: Float
            if logitSoftcap > 0 {
                let capped = logitSoftcap * tanhf(z / logitSoftcap)
                let cappedPenalized = capped > 0 ? capped / penalty : capped * penalty
                // A saturated negative logit times the penalty can leave the
                // softcap's open interval; clamp inside it so atanh stays
                // finite.
                let limit = logitSoftcap * 0.9999
                let clamped = max(min(cappedPenalized, limit), -limit)
                penalized = logitSoftcap * atanhf(clamped / logitSoftcap)
            } else {
                penalized = z > 0 ? z / penalty : z * penalty
            }
            ptr[i] = Float16(penalized)
        }
    }

    /// Clear the incremental penalty history. The scratch sampler outlives a
    /// single generation, so the caller resets it at the start of each one.
    func resetPenaltyHistory() {
        penaltyFrequency.removeAll(keepingCapacity: true)
        penaltyHistorySeeded = false
    }

    // MARK: - Seed

    /// Deterministic per-position seed when `config.seed != nil` so a fixed seed
    /// reproduces across token positions; clock-derived (non-zero) otherwise.
    /// xorshift64 in the kernel has a fixed point at 0, so we never emit 0.
    static func seedFor(config: GenerationConfig, position: Int) -> UInt64 {
        if let s = config.seed {
            let mixed = Self.splitmix64(s &+ UInt64(bitPattern: Int64(position)))
            return mixed == 0 ? 0x9E3779B97F4A7C15 : mixed
        }
        var t = timespec()
        clock_gettime(CLOCK_MONOTONIC, &t)
        // Combine the clock with a monotonic counter (R34) so two samples in
        // the same nanosecond still draw distinct seeds.
        let counter = Self.nondeterministicSeedCounter
            .wrappingAdd(1, ordering: .relaxed).newValue
        let raw = (UInt64(bitPattern: Int64(t.tv_nsec)) &+ counter) &* 0x9E3779B97F4A7C15
            &+ UInt64(bitPattern: Int64(t.tv_sec))
        return raw == 0 ? 0x9E3779B97F4A7C15 : raw
    }

    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
