import Foundation

/// A pinned upstream checkpoint the installer knows how to repack. Each value
/// fixes the repo, revision and index fingerprint so installs are exactly
/// reproducible.
public struct SupportedModelSource: Sendable, Equatable {
    /// CLI selector value (`--model <name>`).
    public let name: String
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    /// Value recorded as `manifest.modelID` when the source fingerprint matches.
    public let modelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }

    /// Download estimate covers the `language_model.*` tensors plus tokenizer
    /// and metadata sidecars. Installed bytes add the resident index and
    /// per-expert 16 KB page rounding plus layout/manifest sidecars.
    public static let qwen36 = SupportedModelSource(
        name: "qwen36",
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        modelID: "qwen3.6-35b-a3b-4bit",
        approximateDownloadBytes: 19_529_025_048,
        installedBytes: 19_546_491_213,
        reserveBytes: 1_073_741_824)

    public static let qwen36_8bit = SupportedModelSource(
        name: "qwen36-8bit",
        displayName: "Qwen3.6 35B-A3B 8-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-8bit",
        revision: "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
        sourceIndexSHA256:
            "3db12edeebeb65cab9a6eeb63cd74be4e0c74139a75f672701290b98230501cf",
        modelID: "qwen3.6-35b-a3b-8bit",
        approximateDownloadBytes: 37_741_392_345,
        installedBytes: 37_800_000_000,
        reserveBytes: 1_073_741_824)

    /// Text-only Ornith 1.5 uses the same qwen3_5_moe tensor contract as the
    /// Qwen 3.6 target. The official MLX conversion intentionally contains
    /// only `language_model.*` tensors; vision and MTP are separate follow-up
    /// features rather than silently entering the text runtime.
    public static let ornith15 = SupportedModelSource(
        name: "ornith15",
        displayName: "Ornith 1.5 35B-A3B 4-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        revision: "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
        sourceIndexSHA256:
            "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36",
        modelID: "ornith-1.5-35b-a3b-4bit",
        approximateDownloadBytes: 19_528_995_943,
        installedBytes: 19_530_000_000,
        reserveBytes: 1_073_741_824)

    public static let ornith15_8bit = SupportedModelSource(
        name: "ornith15-8bit",
        displayName: "Ornith 1.5 35B-A3B 8-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
        revision: "02440c39bdf7365c494a7f55f2a8b104ba87562f",
        sourceIndexSHA256:
            "83c641a791aa957df7d280eef1b0c8faf7a2ec9b19dd3355fb13abae8ae0ed15",
        modelID: "ornith-1.5-35b-a3b-8bit",
        approximateDownloadBytes: 36_848_194_663,
        installedBytes: 36_850_000_000,
        reserveBytes: 1_073_741_824)

    /// One-layer native Qwen3.6 MTP draft. It contains no embedding or LM
    /// head; the runtime reuses those tensors from whichever 4/6/8-bit target
    /// is loaded. The routed experts remain SSD-streamed in `.gturbo` form.
    public static let qwen36MTP = SupportedModelSource(
        name: "qwen36-mtp",
        displayName: "Qwen3.6 35B-A3B native MTP draft 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
        revision: "0295b81421bf4d0fccca9a7c0fcfb1418dda3516",
        sourceIndexSHA256:
            "00e220ddb21ceeb6290a3a1161f97339c553f3d27fc4319900a96edb5cfae74c",
        modelID: "qwen3.6-35b-a3b-mtp-4bit",
        approximateDownloadBytes: 475_130_833,
        installedBytes: 475_300_000,
        reserveBytes: 536_870_912)

    /// Default source when no `--model` selector is given.
    public static let `default` = qwen36

    public static let all: [SupportedModelSource] = [
        qwen36, qwen36_8bit, ornith15, ornith15_8bit, qwen36MTP,
    ]

    public static func named(_ name: String) -> SupportedModelSource? {
        all.first { $0.name == name }
    }
}
