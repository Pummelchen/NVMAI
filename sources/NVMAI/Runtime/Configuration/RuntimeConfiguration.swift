import Foundation

public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

/// Decode scheduling for SSD-backed routed experts.
///
/// `hitFixup` commits phase 1 for resident experts while cache misses are read,
/// then computes only the missed experts before the common reduction. `barrier`
/// preserves the former all-experts-after-I/O path as a correctness/performance
/// control for A/B measurements.
public enum RuntimeDecodeExpertExecution: String, Codable, Sendable {
    case hitFixup = "hit-fixup"
    case barrier
    case gpuResidency = "gpu-residency"

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeDecodeExpertExecution {
        guard let raw = environment["NVMAI_DECODE_EXPERT_EXECUTION"] else {
            return .hitFixup
        }
        guard let value = RuntimeDecodeExpertExecution(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidDecodeExpertExecution(raw)
        }
        return value
    }
}

public enum RuntimeExpertIOSynchronization: String, Codable, Sendable {
    case host
    case event

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeExpertIOSynchronization {
        guard let raw = environment["NVMAI_EXPERT_IO_SYNC"] else { return .host }
        guard let value = RuntimeExpertIOSynchronization(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidExpertIOSynchronization(raw)
        }
        return value
    }
}

public enum RuntimeExpertIOSubmission: String, Codable, Sendable {
    case deferred
    case immediate

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeExpertIOSubmission {
        guard let raw = environment["NVMAI_EXPERT_IO_SUBMISSION"] else { return .deferred }
        guard let value = RuntimeExpertIOSubmission(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidExpertIOSubmission(raw)
        }
        return value
    }
}

/// Storage precision for the autoregressive attention key/value cache.
/// Quantized modes use affine groups of 64 values and keep their scale and
/// bias alongside each token row; model weights are unaffected.
public enum KVCachePrecision: Int, Codable, CaseIterable, Sendable {
    case int4 = 4
    case int8 = 8
    case fp16 = 16

    public var label: String { "\(rawValue)-bit" }
    public var isQuantized: Bool { self != .fp16 }
}

public enum RuntimeRoPEScalingMode: String, Codable, CaseIterable, Sendable {
    case none
    case yarn
}

public enum RuntimeConfigurationError: Error, CustomStringConvertible, Equatable {
    case invalidExpertCacheSlots(Int)
    case invalidPrefillChunkTokens(Int)
    case invalidYaRNContextTokens(Int)
    case contextRequiresYaRN(Int)
    case yaRNContextMismatch(maxContext: Int, configured: Int)
    case yaRNUnsupportedArchitecture
    case invalidDecodeExpertExecution(String)
    case invalidExpertIOSynchronization(String)
    case invalidExpertIOSubmission(String)

    public var description: String {
        switch self {
        case .invalidExpertCacheSlots(let value):
            return "unsupported expert-cache slot count \(value); allowed: \(RuntimeConfiguration.allowedExpertCacheSlots)"
        case .invalidPrefillChunkTokens(let value):
            return "unsupported prefill chunk size \(value); allowed: \(RuntimeConfiguration.allowedPrefillChunkTokens)"
        case .invalidYaRNContextTokens(let value):
            return "unsupported YaRN context \(value); allowed: \(RuntimeConfiguration.supportedYaRNContextTokens)"
        case .contextRequiresYaRN(let value):
            return "context \(value) exceeds the native \(RuntimeConfiguration.nativeMaximumContextTokens)-token limit; enable YaRN"
        case .yaRNContextMismatch(let maxContext, let configured):
            return "YaRN is configured for \(configured) tokens, but max context is \(maxContext)"
        case .yaRNUnsupportedArchitecture:
            return "YaRN requires the Qwen3.5-MoE NeoX sub-dimension RoPE architecture"
        case .invalidDecodeExpertExecution(let value):
            return "unsupported decode expert execution '\(value)'; allowed: hit-fixup, barrier, gpu-residency"
        case .invalidExpertIOSynchronization(let value):
            return "unsupported expert I/O synchronization '\(value)'; allowed: host, event"
        case .invalidExpertIOSubmission(let value):
            return "unsupported expert I/O submission '\(value)'; allowed: deferred, immediate"
        }
    }
}

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let supportedContextTokens = [
        4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,
    ]
    public static let nativeMaximumContextTokens = 262_144
    public static let supportedYaRNContextTokens = [524_288, 1_048_576]
    public static let defaultYaRNContextTokens = 1_048_576
    public static let maximumContextTokens = 1_048_576
    // 112 exists because the useful range ends between 96 and 128 on a 24 GiB
    // machine: 96 measured an 85.4% hit rate, 128 reaches 89.8% but needs
    // ~17 GB of cache and swaps, costing 68%. Without a value in between
    // there was no way to ask whether the extra hit rate is reachable.
    public static let allowedExpertCacheSlots = [8, 16, 24, 32, 40, 48, 64, 96, 112, 128]

    /// Target bytes for the routed-expert slot cache when no count is given.
    ///
    /// 8 GiB, which is a third of a 24 GB machine and deliberate. The slot cache
    /// has to hold the routing working set, and a routing trace over 383 real
    /// tokens measured **131 distinct experts per layer** across a 128-token
    /// window. 128 slots is the first budget that holds it.
    ///
    /// Because expert reads bypass the page cache (see `ParallelExpertReader`),
    /// there is no second cache to fall back on: whatever the slots do not hold is
    /// fetched from SSD every token. That makes the curve a cliff rather than a
    /// slope. Measured, 4-bit, short prompt, bounded:
    ///
    ///      16 slots  1.05 GB   8.73 tok/s   io 49.4 ms
    ///      32 slots  2.11 GB   8.94         io 41.3
    ///      64 slots  4.22 GB   9.91         io 28.3
    ///     128 slots  8.44 GB  18.91         io  7.2
    ///
    /// A smaller budget does not trade throughput gently for memory -- it falls off
    /// by 2.2x while saving RAM that the OS would otherwise have to hold anyway.
    ///
    /// This inverts under the page-cache policy, where the OS holds the working set
    /// and slot memory is redundant pressure: 4-bit measured 13.61 tok/s at 16
    /// slots against 8.78 at 128. So this constant is only correct while expert
    /// reads bypass the cache. Re-tune it if that ever changes, and re-tune it at
    /// the shipped `--max-context`, never a reduced one.
    public static let defaultExpertCacheBudgetBytes = 8 << 30

    /// Decode defaults that are not one number across the catalogue.
    ///
    /// Both settings below are governed by the same quantity: how much of the
    /// token is expert I/O. A speculative read is only ever a bet that the SSD
    /// has service to spare, so it pays where I/O dominates and costs where it
    /// does not.
    public struct DecodeTuning: Sendable, Equatable {
        /// Target bytes for the routed-expert slot cache.
        public let expertCacheBudgetBytes: Int
        /// Speculative expert reads allowed in flight; 0 disables prefetch.
        public let prefetchDepth: Int

        public init(expertCacheBudgetBytes: Int, prefetchDepth: Int) {
            self.expertCacheBudgetBytes = expertCacheBudgetBytes
            self.prefetchDepth = prefetchDepth
        }
    }

    /// The measured optimum for a family at a given routed-expert width.
    ///
    /// Measured 2026-08-30, 512-token continuous prose, fresh process per run,
    /// configs interleaved with the order reversed on alternate repetitions:
    ///
    ///     qwen38flash 4-bit   5.735 -> 6.957 tok/s  (+21.3%)  12 GiB + depth 1
    ///     qwen36      8-bit  11.994 -> 12.659       ( +5.5%)   8 GiB + depth 1
    ///     ornith      8-bit  11.173 -> 11.837       ( +5.9%)   8 GiB + depth 1
    ///     qwen36/ornith 4-bit                        (regress)  8 GiB, no prefetch
    ///
    /// The 4-bit 35B pair is the instructive one. Expert I/O there is only
    /// ~7 ms of a ~44 ms token, so there is almost nothing for a speculative
    /// read to recover, and the read still costs SSD service and ring
    /// bookkeeping -- it measured a regression at 7 runs per config. Ornith and
    /// Qwen 3.6 share the `qwen36` family and measured the same, so keying on
    /// family rather than model id is correct here rather than merely
    /// convenient.
    public static func decodeTuning(family: ModelFamily,
                                    weightBits: Int) -> DecodeTuning {
        switch (family, weightBits) {
        case (.qwen38flash, _), (.qwen38flashMTP, _):
            // 96 slots. The only family whose working set justifies the extra
            // 4 GiB: 512 experts at top-10 spread far wider than 128 at top-8,
            // so it is still climbing at 96 where the 35B families have
            // flattened above 90% hit rate.
            return DecodeTuning(expertCacheBudgetBytes: 12 << 30, prefetchDepth: 1)
        case (.qwen36, 8), (.qwen36MTP, 8):
            return DecodeTuning(expertCacheBudgetBytes: defaultExpertCacheBudgetBytes,
                                prefetchDepth: 1)
        default:
            return DecodeTuning(expertCacheBudgetBytes: defaultExpertCacheBudgetBytes,
                                prefetchDepth: 0)
        }
    }

    /// A tuned budget the machine can actually hold.
    ///
    /// `decodeTuning` returns what measured fastest on a 24 GiB machine. Half
    /// of physical memory is the ceiling because the slot cache is not the only
    /// resident claim -- dense weights, the KV cache and the prompt cache all
    /// have to fit beside it. Without this, a 12 GiB default aimed at
    /// qwen38flash would be handed unchanged to a 16 GiB Mac.
    public static func affordableExpertCacheBudget(
        _ wanted: Int,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Int {
        guard physicalMemory > 0 else { return wanted }
        return min(wanted, Int(physicalMemory / 2))
    }

    /// Parses a RAM budget such as `2G`, `512M`, `8GiB` or a plain byte count.
    ///
    /// Accepts the sizes users actually type. Returns nil for anything
    /// unparseable or non-positive, so a typo becomes an argument error rather
    /// than a silently tiny cache.
    public static func parseBudgetBytes(_ text: String) -> Int? {
        let raw = text.trimmingCharacters(in: .whitespaces).uppercased()
        guard !raw.isEmpty else { return nil }
        let multipliers: [(String, Int)] = [
            ("GIB", 1 << 30), ("MIB", 1 << 20), ("KIB", 1 << 10),
            ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10),
            ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10),
        ]
        for (suffix, scale) in multipliers where raw.hasSuffix(suffix) {
            let number = String(raw.dropLast(suffix.count))
                .trimmingCharacters(in: .whitespaces)
            guard let value = Double(number), value > 0 else { return nil }
            let bytes = value * Double(scale)
            guard bytes.isFinite, bytes >= 1, bytes < Double(Int.max) else { return nil }
            return Int(bytes)
        }
        guard let plain = Int(raw), plain > 0 else { return nil }
        return plain
    }

    /// Slots that fit `budgetBytes`, snapped to the nearest supported count.
    ///
    /// Deriving from the stride rather than hard-coding a number per quantisation
    /// keeps 4-bit and 8-bit on the same rule: 1 GiB lands on 16 slots at a
    /// 1.688 MiB stride and 8 slots at 3.188 MiB, which are the measured optima
    /// for each.
    public static func expertCacheSlots(
        expertStrideBytes: UInt64,
        layers: Int,
        budgetBytes: Int = defaultExpertCacheBudgetBytes) -> Int {
        guard expertStrideBytes > 0, layers > 0 else {
            return allowedExpertCacheSlots.first ?? 8
        }
        let perSlot = Double(expertStrideBytes) * Double(layers)
        let wanted = Double(budgetBytes) / perSlot
        var choice = allowedExpertCacheSlots.min {
            abs(Double($0) - wanted) < abs(Double($1) - wanted)
        } ?? allowedExpertCacheSlots.first ?? 8
        // Nearest, then step down until the footprint honours the budget.
        //
        // Rounding to nearest alone can overshoot, and the overshoot grows with
        // the expert stride: the wider the model, the further past the budget
        // the nearest rung lands. Qwen3.8 at 8-bit wanted 51.4 slots and was
        // handed 64 -- a 16.1 GiB cache against a 12 GiB budget, 34% over. On a
        // 24 GiB machine that put the process into swap and cost 4.8x
        // throughput: 0.42 tok/s at 64 slots against 2.01 at 32, with the hit
        // rate *falling* 80% -> 67.7%, which is how a paging problem looks when
        // it is mistaken for a cache problem. The same arithmetic at 4-bit
        // wants 96.9 and gets 96, so it never showed on the model this was
        // tuned against.
        //
        // 1.15 is not a new number: `chosenCountStaysNearTheRequestedBudget`
        // has always asserted the footprint stays within 15% of the budget. It
        // simply never sampled a 48-layer model at a 12 GiB budget, so the one
        // configuration that broke the contract went unmeasured.
        //
        // Stepping down is the safe direction. The measured cache curve is flat
        // below the RAM limit -- 112 slots cut 4-bit's I/O time 12% for no
        // throughput at all -- and a cliff above it. Too few slots costs a
        // little; too many costs everything.
        let ceiling = Double(budgetBytes) * 1.15
        while Double(choice) * perSlot > ceiling,
              let smaller = allowedExpertCacheSlots.last(where: { $0 < choice }) {
            choice = smaller
        }
        return choice
    }
    public static let allowedPrefillChunkTokens = [
        32, 64, 128, 256, 512, 1_024, 2_048, 4_096,
    ]
    public static let qwenLongPrefillChunkTokens = 4_096

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    public let decodeExpertExecution: RuntimeDecodeExpertExecution
    public let expertIOSynchronization: RuntimeExpertIOSynchronization
    public let expertIOSubmission: RuntimeExpertIOSubmission
    public let kvCachePrecision: KVCachePrecision
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let yarnContextTokens: Int

    public init(expertCacheSlots: Int = 64,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .default,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                decodeExpertExecution: RuntimeDecodeExpertExecution = .hitFixup,
                expertIOSynchronization: RuntimeExpertIOSynchronization = .host,
                expertIOSubmission: RuntimeExpertIOSubmission = .deferred,
                kvCachePrecision: KVCachePrecision = .int8,
                ropeScalingMode: RuntimeRoPEScalingMode = .none,
                yarnContextTokens: Int = RuntimeConfiguration.defaultYaRNContextTokens) throws {
        guard Self.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw RuntimeConfigurationError.invalidExpertCacheSlots(expertCacheSlots)
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw RuntimeConfigurationError.invalidPrefillChunkTokens(prefillChunkTokens)
        }
        guard Self.supportedYaRNContextTokens.contains(yarnContextTokens) else {
            throw RuntimeConfigurationError.invalidYaRNContextTokens(yarnContextTokens)
        }
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.decodeExpertExecution = decodeExpertExecution
        self.expertIOSynchronization = expertIOSynchronization
        self.expertIOSubmission = expertIOSubmission
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
        self.yarnContextTokens = yarnContextTokens
    }

    public func validate(maxContext: Int) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        switch ropeScalingMode {
        case .none:
            guard maxContext <= Self.nativeMaximumContextTokens else {
                throw RuntimeConfigurationError.contextRequiresYaRN(maxContext)
            }
        case .yarn:
            guard maxContext == yarnContextTokens else {
                throw RuntimeConfigurationError.yaRNContextMismatch(
                    maxContext: maxContext, configured: yarnContextTokens)
            }
        }
    }

    public static var production: RuntimeConfiguration {
        // lint:allow-force every default is a compile-time constant on the
        // allowed lists, so the validating init cannot throw here;
        // RuntimeConfigurationTests pins that.
        try! RuntimeConfiguration()
    }

    /// Production pins the sliding-window ring on. This is deliberately a
    /// constant and not a stored option: `KVCacheManager` takes the flag as a
    /// real parameter (tests construct it both ways to cover the non-ring
    /// path), but the shipping runtime has exactly one supported setting, and
    /// the value is part of `ServerPromptCacheDomain` — making it settable
    /// would let two processes disagree about the layout of a persisted KV
    /// snapshot. Read-only here is the guarantee, not an oversight.
    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
