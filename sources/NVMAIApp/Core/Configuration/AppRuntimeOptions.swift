import Foundation
import NVMAI

public enum AppExpertCachePolicy: String, CaseIterable, Sendable, Identifiable {
    case lfu
    case lru

    public var id: String { rawValue }
    public var label: String { rawValue.uppercased() }
}

public enum AppRDAdvicePolicy: String, CaseIterable, Sendable, Identifiable {
    case off
    case `default`
    case bounded
    case adaptive

    public var id: String { rawValue }
    public var label: String { rawValue.capitalized }

    var runtimeValue: RDAdvicePolicyMode {
        switch self {
        case .off: return .off
        case .default: return .default
        case .bounded: return .bounded
        case .adaptive: return .adaptive
        }
    }
}

public enum AppModelVerification: String, CaseIterable, Sendable, Identifiable {
    case fullSha256 = "full-sha256"
    case trustedInstall = "trusted-install"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .fullSha256: return "Full SHA-256"
        case .trustedInstall: return "Trust verified install"
        }
    }

    var runtimeValue: ModelIntegrityPolicy {
        switch self {
        case .fullSha256: return .fullSha256
        case .trustedInstall: return .sizeCheckTrustedReceipt
        }
    }
}

public struct AppRuntimeOptions: Equatable, Sendable {
    public static let allowedSlotCounts = RuntimeConfiguration.allowedExpertCacheSlots
    /// `expertCacheSlots` value meaning "size the cache from the model's
    /// tuning profile at load": the same budget-to-slots resolution the
    /// server and the CLI perform, so the app runs every install at its
    /// measured optimum instead of a flat count. The default.
    public static let automaticSlotCount = 0
    public static let allowedPrefillChunkTokens =
        RuntimeConfiguration.allowedPrefillChunkTokens

    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification
    public var conciseMode: Bool
    public var thinkingMode: ModelThinkingMode
    public var kvCachePrecision: KVCachePrecision
    public var ropeScalingMode: RuntimeRoPEScalingMode

    public init(expertCacheSlots: Int = automaticSlotCount,
                expertCachePolicy: AppExpertCachePolicy = .lfu,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = RuntimeConfiguration.qwenLongPrefillChunkTokens,
                rdadvisePolicy: AppRDAdvicePolicy = .default,
                modelVerification: AppModelVerification = .fullSha256,
                conciseMode: Bool = false,
                thinkingMode: ModelThinkingMode = .off,
                kvCachePrecision: KVCachePrecision = .int8,
                ropeScalingMode: RuntimeRoPEScalingMode = .none) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
        self.conciseMode = conciseMode
        self.thinkingMode = thinkingMode
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
    }

    public func validate() throws {
        guard expertCacheSlots == Self.automaticSlotCount
                || Self.allowedSlotCounts.contains(expertCacheSlots) else {
            throw AppInferenceError.invalidRequest(
                "expert cache slots must be one of \(Self.allowedSlotCounts)")
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw AppInferenceError.invalidRequest(
                "prefill chunk size must be one of \(Self.allowedPrefillChunkTokens)")
        }
    }

    public var prefillConfig: PrefillRuntimeConfig {
        prefillEnabled ? .production(chunkTokens: prefillChunkTokens) : .off
    }

    public var resultSummary: String {
        let prefill = prefillEnabled ? "prefill \(prefillChunkTokens)" : "prefill off"
        let verification = modelVerification == .fullSha256 ? "full SHA-256" : "trusted receipt"
        let scaling = ropeScalingMode == .yarn ? "YaRN" : "native RoPE"
        let cache = expertCacheSlots == Self.automaticSlotCount ? "auto" : "\(expertCacheSlots)"
        return "Cache \(cache) \(expertCachePolicy.label), \(prefill), \(kvCachePrecision.label) KV, \(scaling), thinking \(thinkingMode.rawValue), RDADVISE \(rdadvisePolicy.label.lowercased()), \(verification)"
    }

    public static func slotsLabel(for slots: Int) -> String {
        // D21: the per-slot memory deltas were hardcoded and drifted from the
        // real pack layout, so the picker shows names only; the memory
        // implications are described in the UI text next to the picker.
        slots == automaticSlotCount ? "Auto (model profile)" : "\(slots)"
    }

    /// The slot count the model's tuning profile recommends on this machine:
    /// the (model, width) row's budget, clamped to half of physical memory,
    /// turned into slots from the manifest's expert stride -- exactly what
    /// NVMAIServer and NVMAICLI do without `--ram-budget`. Falls back to the
    /// production default when the manifest cannot be read; the load that
    /// follows reports the real error.
    public static func recommendedSlots(forModelAt directory: URL) -> Int {
        guard let identity = try? ManifestReader.peekIdentity(directoryURL: directory),
              let arch = ArchConfig.knownArchitectures[identity.family],
              let manifest = try? ManifestReader.load(directoryURL: directory, expecting: arch)
        else { return RuntimeConfiguration.production.expertCacheSlots }
        let budget = RuntimeConfiguration.affordableExpertCacheBudget(
            ModelProfile.resolve(identity: identity).expertCacheBudgetBytes)
        return RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: manifest.expertStride,
            layers: manifest.arch.numLayers,
            budgetBytes: budget)
    }

    /// The slot count this configuration loads with: the explicit choice, or
    /// the profile's recommendation for the model at `modelDirectory`.
    public func effectiveSlots(forModelAt modelDirectory: URL?) -> Int {
        guard expertCacheSlots == Self.automaticSlotCount else { return expertCacheSlots }
        guard let modelDirectory else { return RuntimeConfiguration.production.expertCacheSlots }
        return Self.recommendedSlots(forModelAt: modelDirectory)
    }

    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool,
        maxContextTokens: Int = RuntimeConfiguration.defaultYaRNContextTokens,
        modelDirectory: URL? = nil
    ) throws -> RuntimeConfiguration {
        try validate()
        return try RuntimeConfiguration(
            expertCacheSlots: effectiveSlots(forModelAt: modelDirectory),
            expertCachePolicy: expertCachePolicy == .lru ? .lru : .lfu,
            rdadvisePolicy: rdadvisePolicy.runtimeValue,
            prefillEnabled: prefillEnabled,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode,
            yarnContextTokens: ropeScalingMode == .yarn
                ? maxContextTokens : RuntimeConfiguration.defaultYaRNContextTokens)
    }
}

public struct AppLoadedRuntimeKey: Equatable, Sendable {
    public var modelDirectory: URL
    public var maxContextTokens: Int
    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification
    public var forceLogitsHead: Bool
    public var kvCachePrecision: KVCachePrecision
    public var ropeScalingMode: RuntimeRoPEScalingMode
    public var thinkingMode: ModelThinkingMode

    public init(modelDirectory: URL,
                maxContextTokens: Int,
                options: AppRuntimeOptions,
                forceLogitsHead: Bool = false) {
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.maxContextTokens = maxContextTokens
        self.expertCacheSlots = options.expertCacheSlots
        self.expertCachePolicy = options.expertCachePolicy
        self.rdadvisePolicy = options.rdadvisePolicy
        self.modelVerification = options.modelVerification
        self.forceLogitsHead = forceLogitsHead
        self.kvCachePrecision = options.kvCachePrecision
        self.ropeScalingMode = options.ropeScalingMode
        self.thinkingMode = options.thinkingMode
    }
}
