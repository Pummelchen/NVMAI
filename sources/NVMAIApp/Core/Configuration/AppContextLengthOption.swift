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

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        "\(tokens / 1_024)K"
    }

    public var fp16KVBytes: UInt64 {
        let architecture = ArchConfig.qwen36_35B_A3B
        // Qwen uses 2 for Gated DeltaNet layers. Those layers keep recurrent
        // state rather than allocating the attention KV ring estimated here.
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 1 ? 1 : 0)
        }
        let slidingLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 1 : 0)
        }
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + PrefillRuntimeConfig.defaultChunked.chunkTokens)
        let slidingBytesPerRow = architecture.numKVHeads
            * architecture.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = architecture.numFullKVHeads
            * architecture.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    public var menuLabel: String {
        // D20: derive the memory delta from the computed FP16 KV footprint
        // (relative to the 4K baseline) so it cannot drift from the real
        // per-token byte math.
        let baseline = AppContextLengthOption.fourK.fp16KVBytes
        guard fp16KVBytes >= baseline else { return shortLabel }
        let delta = Int64(fp16KVBytes) - Int64(baseline)
        guard delta > 0 else { return "\(shortLabel), Default" }
        return "\(shortLabel), +\(Self.memoryDelta(delta))"
    }

    private static func memoryDelta(_ bytes: Int64) -> String {
        let locale = Locale(identifier: "en_US_POSIX")
        let mebibytes = Double(bytes) / 1_048_576
        if mebibytes < 1_024 {
            return String(format: "%.0f MB", locale: locale, mebibytes)
        }
        return String(format: "%.2f GB", locale: locale, mebibytes / 1_024)
    }
}
