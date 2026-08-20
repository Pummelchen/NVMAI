import Foundation
import NVMAI

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384
    case thirtyTwoK = 32_768
    case sixtyFourK = 65_536
    case oneTwentyEightK = 131_072
    case twoFiftySixK = 262_144
    case fiveTwelveK = 524_288
    case oneM = 1_048_576

    public static let nativeCases: [Self] = [
        .fourK, .eightK, .sixteenK, .thirtyTwoK, .sixtyFourK,
        .oneTwentyEightK, .twoFiftySixK,
    ]
    public static let yarnCases: [Self] = [.fiveTwelveK, .oneM]

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        tokens == 1_048_576 ? "1M" : "\(tokens / 1_024)K"
    }

    public func kvBytes(precision: KVCachePrecision) -> UInt64 {
        let architecture = ArchConfig.qwen36_35B_A3B
        // Qwen uses 2 for Gated DeltaNet layers. Those layers keep recurrent
        // state rather than allocating the attention KV ring estimated here.
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 1 ? 1 : 0)
        }
        let slidingLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 1 : 0)
        }
        let keyAndValue = 2
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + PrefillRuntimeConfig.defaultChunked.chunkTokens)
        func rowBytes(elements: Int) -> Int {
            if precision == .fp16 { return elements * 2 }
            let values = (elements * precision.rawValue + 7) / 8
            let alignedValues = (values + 1) & ~1
            let groups = (elements + KVCacheManager.quantizationGroupSize - 1)
                / KVCacheManager.quantizationGroupSize
            return alignedValues + groups * 4
        }
        let slidingBytesPerRow = rowBytes(
            elements: architecture.numKVHeads * architecture.headDim) * keyAndValue
        let fullBytesPerRow = rowBytes(
            elements: architecture.numFullKVHeads * architecture.fullHeadDim) * keyAndValue
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    public var fp16KVBytes: UInt64 { kvBytes(precision: .fp16) }

    public func menuLabel(precision: KVCachePrecision) -> String {
        // D20: derive the memory delta from the selected KV footprint
        // (relative to the 4K baseline) so it cannot drift from the real
        // per-token byte math.
        let bytes = kvBytes(precision: precision)
        let baseline = AppContextLengthOption.fourK.kvBytes(precision: precision)
        guard bytes >= baseline else { return shortLabel }
        let delta = Int64(bytes) - Int64(baseline)
        guard delta > 0 else { return "\(shortLabel), Default" }
        return "\(shortLabel), +\(Self.memoryDelta(delta))"
    }

    public var menuLabel: String { menuLabel(precision: .fp16) }

    private static func memoryDelta(_ bytes: Int64) -> String {
        let locale = Locale(identifier: "en_US_POSIX")
        let mebibytes = Double(bytes) / 1_048_576
        if mebibytes < 1_024 {
            return String(format: "%.0f MB", locale: locale, mebibytes)
        }
        return String(format: "%.2f GB", locale: locale, mebibytes / 1_024)
    }
}
