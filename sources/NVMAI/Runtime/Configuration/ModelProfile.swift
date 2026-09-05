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
    /// Disk I/O policy for those reads: 0 is the default tier, otherwise an
    /// IOPOL_* value (IOPOL_UTILITY measured +3% at depth 2 on Qwen 3.8 4-bit;
    /// IOPOL_THROTTLE measured a loss).
    public var prefetchIOTier: Int32
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
    /// Compute phase 1 for the GPU-classified resident experts in a command
    /// buffer queued right behind the router, instead of after the CPU's
    /// cache plan. Needs the GPU residency classifier and the pooled cache
    /// layout, both of which the runner selects when this is set. Measured
    /// on Qwen3.8 4-bit: 6.02 / 6.10 tok/s against 5.82 / 5.96, GPU wait
    /// -3.5 ms/token, output byte-identical.
    public var earlyExpertHits: Bool
    /// Keep the routed-expert cache wired through prefill instead of
    /// unpinning it at prefill start and re-wiring it on the first decode
    /// token. Measured on Qwen3.8 4-bit: the re-wire faults a swapped-out
    /// 12 GiB cache back in, 1.6-4.7 s per request; holding it is a wash on
    /// decode throughput and prefill time. NVMAI_KEEP_WIRED=1 forces it on.
    public var keepExpertCacheWired: Bool

    /// The shipped entries. Measured values, each on its own install.
    public static let table: [Key: (budget: Int, prefetch: Int, tier: Int32, chunk: Int?,
                                    sampling: GenerationDefaults.Sampling,
                                    topKSimd: Bool, attnSimd: Bool,
                                    hcFused: Bool, qsaSelect: Bool, keepWired: Bool,
                                    earlyHits: Bool)] = [
        // The 35B rows take prefetch depth 1 and hold the expert cache wired
        // through prefill (2026-09-05, on the repaired ring). Measured per
        // install, five interleaved pairs on Qwen 3.6 and three on the other
        // two, 512-token generations:
        //
        //   prefetch depth 1   4-bit            8-bit
        //     Qwen 3.6         +1.8%            +11.3% (+9.8% on a rerun)
        //     Ornith 1.5       +1.8%            +12.6%
        //     AgentWorld       +1.4%            +11.4%
        //
        // Every interval excludes zero. Depth 2 gives only +2.7% where depth 1
        // gives +9.8%, so the ring stays one read deep, as it was at 4.7.
        // The 8-bit gain is the one the clogged ring had been hiding since
        // 2026-09-04: those rows asked for depth 1 and got nothing.
        //
        // Keeping the cache wired is worth +0.9% / +1.6% on 512-token
        // generations and +7.1% / +5.1% on 48-token ones (Qwen 3.6 4/8-bit),
        // where the first decode token no longer faults a swapped-out 10-12
        // GiB cache back in. Measured on Qwen 3.6; the other two installs
        // have identical geometry and cache sizes and take it by inference.
        // Qwen 3.6 35B-A3B, slot A/B 2026-09-05 (essay, interleaved, swap
        // sampled): 4-bit 128 slots 19.50 / 20.45, 160 (10 GiB) 20.55 / 21.04
        // with swap flat, 192 (12 GiB) 21.61 / 21.60 but 1.5 GB pushed to swap
        // on first contact -- 160 is the 24 GB default, 192 is one budget
        // setting away. 8-bit 64 slots 9.85 / 9.72, 96 (12 GiB) 11.13 / 11.19,
        // swap flat: the 8-bit hit rate was 79% at 64 (37% of the token in
        // exposed expert reads) and the trace simulation halves the misses
        // at 96. Prefetch one deep still pays at 8-bit; utility-tier depth 2
        // measured a wash at both widths.
        Key("qwen3.6-35b-a3b", 4): (10 << 30, 1, 0, 4_096, GenerationDefaults.house, true, true, false, false, true, false),
        Key("qwen3.6-35b-a3b", 8): (12 << 30, 1, 0, 4_096, GenerationDefaults.house, true, true, false, false, true, false),
        // Ornith 1.5, same geometry, measured on its own 2026-09-05: 4-bit
        // 128 slots 19.91 / 20.41 vs 160 20.84 / 21.02; 8-bit 64 slots
        // 8.69 / 9.12 vs 96 10.83 / 10.86, swap flat on every arm.
        Key("ornith-1.5-35b-a3b", 4): (10 << 30, 1, 0, 4_096, GenerationDefaults.house, true, true, false, false, true, false),
        Key("ornith-1.5-35b-a3b", 8): (12 << 30, 1, 0, 4_096, GenerationDefaults.house, true, true, false, false, true, false),
        // AgentWorld, measured on its own 2026-09-05: 4-bit 128 slots 20.52 /
        // 20.50 vs 160 21.11 / 20.92; 8-bit 64 slots 9.31 / 9.25 vs 96
        // 11.15 / 11.21, swap flat. Residency and barrier execution both
        // lose on it.
        Key("qwen-agentworld", 4): (10 << 30, 1, 0, 4_096, GenerationDefaults.house, true, true, false, false, true, false),
        Key("qwen-agentworld", 8): (12 << 30, 1, 0, 4_096, GenerationDefaults.house, true, true, false, false, true, false),
        // Qwen3.8-Flash-Next: 96 slots (12 GiB) still climbing, prefetch one
        // deep +12%; its card specifies temperature 1.0 / top-p 0.95. The
        // fused hyper-connection gates and the GPU key select are measured
        // washes and stay off.
        // 4-bit, 2026-09-05 profile: prefetch OFF. Per-token counters showed
        // the ring had been clogged since its first token (a wrong prediction
        // for layer 47 was never reclaimed), so every earlier prefetch
        // measurement compared variants of a mechanism that issued ~5 reads
        // per token. Repaired and running as designed it loses on this SSD at
        // every setting (512-token story runs, prefetch off 5.81 / 5.83; rank
        // order 4.70 / 4.66; probe-margin gate 0.03 4.95 / 5.26; 0.06 5.10 /
        // 5.41): a one-layer-ahead read lands after the next plan and takes
        // SSD time from the demand reads. The cache stays wired through
        // prefill (see keepExpertCacheWired).
        Key("qwen3.8-flash-next", 4): (12 << 30, 0, 0, 4_096,
                                       GenerationDefaults.Sampling(temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                                       true, true, false, false, true, false),
        // 8-bit: 32 slots (8 GiB) 2.05 / 2.06 tok/s; 40 slots (9.5 GiB) 2.18 /
        // 2.27 with swap falling; 48 (13 GiB) 2.24-2.33 but ~1 GB of swap
        // growth per run on this 24 GB machine. 40 is the no-paging middle.
        Key("qwen3.8-flash-next", 8): (Int(9.5 * Double(1 << 30)), 0, 0, 4_096,
                                       GenerationDefaults.Sampling(temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                                       true, true, false, false, true, false),
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
            prefetchIOTier: 0,
            prefillChunkTokens: nil,
            sampling: GenerationDefaults.forFamily(family),
            routerTopKSimd: true, attentionSimdPartial: true,
            hcFused: false, qsaGPUSelect: false, earlyExpertHits: false,
            keepExpertCacheWired: false)
        if let row = table[profile.key] {
            profile.expertCacheBudgetBytes = row.budget
            profile.prefetchDepth = row.prefetch
            profile.prefetchIOTier = row.tier
            profile.prefillChunkTokens = row.chunk
            profile.sampling = row.sampling
            profile.routerTopKSimd = row.topKSimd
            profile.attentionSimdPartial = row.attnSimd
            profile.hcFused = row.hcFused
            profile.qsaGPUSelect = row.qsaSelect
            profile.keepExpertCacheWired = row.keepWired
            profile.earlyExpertHits = row.earlyHits
        }
        if let v = env["NVMAI_ROUTER_TOPK_SIMD"] { profile.routerTopKSimd = v != "0" }
        if let v = env["NVMAI_ATTN_SIMD_PARTIAL"] { profile.attentionSimdPartial = v != "0" }
        if let v = env["NVMAI_HC_FUSED"] { profile.hcFused = v == "1" }
        if let v = env["NVMAI_QSA_GPU_SELECT"] { profile.qsaGPUSelect = v == "1" || v == "verify" }
        if env["NVMAI_KEEP_WIRED"] == "1" { profile.keepExpertCacheWired = true }
        if let v = env["NVMAI_EARLY_HITS"] { profile.earlyExpertHits = v == "1" }
        if let v = env["NVMAI_PREDICTIVE_PREFETCH"] {
            profile.prefetchDepth = v == "1" ? max(1, profile.prefetchDepth) : 0
        }
        if let v = env["NVMAI_PREFETCH_TOP_M"].flatMap(Int.init), profile.prefetchDepth > 0 {
            profile.prefetchDepth = v
        }
        switch env["NVMAI_PREFETCH_IO_TIER"] {
        case "default": profile.prefetchIOTier = 0
        case "standard": profile.prefetchIOTier = IOPOL_STANDARD
        case "utility": profile.prefetchIOTier = IOPOL_UTILITY
        case "throttle": profile.prefetchIOTier = IOPOL_THROTTLE
        default: break
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
        + "budget=\(expertCacheBudgetBytes >> 20)MiB prefetch=\(prefetchDepth) tier=\(prefetchIOTier) "
        + "chunk=\(prefillChunkTokens.map(String.init) ?? "fallback") "
        + "sampling=\(sampling.temperature)/\(sampling.topK)/\(sampling.topP) "
        + "topk_simd=\(routerTopKSimd) attn_simd=\(attentionSimdPartial) "
        + "hc_fused=\(hcFused) qsa_select=\(qsaGPUSelect) keep_wired=\(keepExpertCacheWired) early_hits=\(earlyExpertHits)"
    }
}
