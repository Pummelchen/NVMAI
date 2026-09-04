import Foundation

/// One tuning profile per (model, routed-expert width): everything the
/// runtime chooses for a model that is not architecture -- the expert-cache
/// budget, the prefetch depth, the prefill chunk, the sampling defaults and
/// the kernel switches -- so each install is tuned on its own and a change
/// to one never touches another.
///
/// Resolution order, last wins:
///   1. the family default (`RuntimeConfiguration.decodeTuning`,
///      `GenerationDefaults.forFamily`, the family's chunk);
///   2. the entry for this (modelID, width) in `ModelProfile.table`;
///   3. the environment switches, so an experiment can still override a
///      shipped value without editing the table.
///
/// The table spells every entry out in full, even where it equals the family
/// default, precisely so that editing one row cannot change another. The
/// resolved profile is logged once at load under NVMAI_RUNNER_STATS.
public struct ModelProfile: Sendable, Equatable {
    public struct Key: Hashable, Sendable {
        public let modelID: String
        public let weightBits: Int
        public init(_ modelID: String, _ weightBits: Int) {
            self.modelID = modelID
            self.weightBits = weightBits
        }
    }

    public let key: Key
    public let family: ModelFamily
    /// Target bytes for the routed-expert slot cache (before the RAM clamp).
    public var expertCacheBudgetBytes: Int
    /// Speculative expert reads in flight; 0 disables prefetch.
    public var prefetchDepth: Int
    /// Prefill chunk in tokens; nil takes the front end's fallback.
    public var prefillChunkTokens: Int?
    public var sampling: GenerationDefaults.Sampling
    /// One-simdgroup router top-k (both k == 8 and k != 8).
    public var routerTopKSimd: Bool
    /// Simdgroup-per-key sparse decode attention (keep-mask layers only).
    public var attentionSimdPartial: Bool
    /// Fused hyper-connection gates (hyper-connection families only).
    public var hcFused: Bool
    /// GPU-side QSA key selection (sparse-attention families only).
    public var qsaGPUSelect: Bool

    /// The shipped entries. Measured values, each on its own install.
    public static let table: [Key: (budget: Int, prefetch: Int, chunk: Int?,
                                    sampling: GenerationDefaults.Sampling,
                                    topKSimd: Bool, attnSimd: Bool,
                                    hcFused: Bool, qsaSelect: Bool)] = [
        // Qwen 3.6 35B-A3B: 128 slots at 4-bit hold the working set; the
        // 8-bit cache is budget-capped to 64 and prefetch one deep pays there.
        Key("qwen3.6-35b-a3b", 4): (8 << 30, 0, 4_096, GenerationDefaults.house, true, true, false, false),
        Key("qwen3.6-35b-a3b", 8): (8 << 30, 1, 4_096, GenerationDefaults.house, true, true, false, false),
        // Ornith 1.5: same geometry, measured the same as Qwen 3.6 in August.
        Key("ornith-1.5-35b-a3b", 4): (8 << 30, 0, 4_096, GenerationDefaults.house, true, true, false, false),
        Key("ornith-1.5-35b-a3b", 8): (8 << 30, 1, 4_096, GenerationDefaults.house, true, true, false, false),
        // AgentWorld: ~91% hit rate at 128 slots against a 93% ceiling, so
        // no budget lever; residency and barrier execution both lose on it.
        Key("qwen-agentworld", 4): (8 << 30, 0, 4_096, GenerationDefaults.house, true, true, false, false),
        Key("qwen-agentworld", 8): (8 << 30, 1, 4_096, GenerationDefaults.house, true, true, false, false),
        // Qwen3.8-Flash-Next: 96 slots (12 GiB) still climbing, prefetch one
        // deep +12%; its card specifies temperature 1.0 / top-p 0.95. The
        // fused hyper-connection gates and the GPU key select are measured
        // washes and stay off.
        Key("qwen3.8-flash-next", 4): (12 << 30, 1, 4_096,
                                       GenerationDefaults.Sampling(temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                                       true, true, false, false),
        Key("qwen3.8-flash-next", 8): (12 << 30, 1, 4_096,
                                       GenerationDefaults.Sampling(temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                                       true, true, false, false),
    ]

    /// Environment switches applied last. Read once per process.
    public static let environment = ProcessInfo.processInfo.environment

    public static func resolve(modelID: String, family: ModelFamily, weightBits: Int,
                               environment env: [String: String] = environment) -> ModelProfile {
        let familyTuning = RuntimeConfiguration.decodeTuning(family: family, weightBits: weightBits)
        var profile = ModelProfile(
            key: Key(modelID, weightBits), family: family,
            expertCacheBudgetBytes: familyTuning.expertCacheBudgetBytes,
            prefetchDepth: familyTuning.prefetchDepth,
            prefillChunkTokens: nil,
            sampling: GenerationDefaults.forFamily(family),
            routerTopKSimd: true, attentionSimdPartial: true,
            hcFused: false, qsaGPUSelect: false)
        if let row = table[profile.key] {
            profile.expertCacheBudgetBytes = row.budget
            profile.prefetchDepth = row.prefetch
            profile.prefillChunkTokens = row.chunk
            profile.sampling = row.sampling
            profile.routerTopKSimd = row.topKSimd
            profile.attentionSimdPartial = row.attnSimd
            profile.hcFused = row.hcFused
            profile.qsaGPUSelect = row.qsaSelect
        }
        if let v = env["NVMAI_ROUTER_TOPK_SIMD"] { profile.routerTopKSimd = v != "0" }
        if let v = env["NVMAI_ATTN_SIMD_PARTIAL"] { profile.attentionSimdPartial = v != "0" }
        if let v = env["NVMAI_HC_FUSED"] { profile.hcFused = v == "1" }
        if let v = env["NVMAI_QSA_GPU_SELECT"] { profile.qsaGPUSelect = v == "1" || v == "verify" }
        if let v = env["NVMAI_PREDICTIVE_PREFETCH"] {
            profile.prefetchDepth = v == "1" ? max(1, profile.prefetchDepth) : 0
        }
        if let v = env["NVMAI_PREFETCH_TOP_M"].flatMap(Int.init), profile.prefetchDepth > 0 {
            profile.prefetchDepth = v
        }
        return profile
    }

    public static func resolve(identity: ManifestIdentity) -> ModelProfile {
        resolve(modelID: identity.modelID, family: identity.family, weightBits: identity.weightBits)
    }

    /// Whether the table names this install, as opposed to falling back to
    /// its family. Draft-head sidecars and unknown ids fall back.
    public var isTabled: Bool { Self.table[key] != nil }

    /// One line for the log, so a benchmark records what ran.
    public var summary: String {
        "profile model=\(key.modelID) bits=\(key.weightBits) family=\(family.rawValue) "
        + (isTabled ? "tabled" : "family-default") + " "
        + "budget=\(expertCacheBudgetBytes >> 20)MiB prefetch=\(prefetchDepth) "
        + "chunk=\(prefillChunkTokens.map(String.init) ?? "fallback") "
        + "sampling=\(sampling.temperature)/\(sampling.topK)/\(sampling.topP) "
        + "topk_simd=\(routerTopKSimd) attn_simd=\(attentionSimdPartial) "
        + "hc_fused=\(hcFused) qsa_select=\(qsaGPUSelect)"
    }
}
