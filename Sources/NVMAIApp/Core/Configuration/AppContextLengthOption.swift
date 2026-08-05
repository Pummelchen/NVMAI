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
        switch self {
        case .fourK: "4K, Default"
        case .eightK: "8K, +80 MB"
        case .sixteenK: "16K, +240 MB"
        case .thirtyTwoK: "32K, +560 MB"
        case .sixtyFourK: "64K, +1.17 GB"
        case .oneTwentyEightK: "128K, +2.42 GB"
        case .twoFiftySixK: "256K, +4.92 GB"
        }
    }
}
