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

    public static let gemma4 = SupportedModelSource(
        name: "gemma4",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        modelID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    /// Download estimate covers the `language_model.*` tensors plus tokenizer
    /// and metadata sidecars; the vision tower is never fetched. Installed
    /// bytes add the resident index and per-expert 16 KB page rounding
    /// (the 1,769,472-byte expert blob is already page-aligned) plus
    /// layout/manifest sidecars.
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

    public static let qwen36_6bit = SupportedModelSource(
        name: "qwen36-6bit",
        displayName: "Qwen3.6 35B-A3B 6-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-6bit",
        revision: "cb7e092ef8efe540bc3672c8929c4adbe5f4f759",
        sourceIndexSHA256:
            "eaea194dfb961e6a5215dcc6e4dd42d0df6efe8d8686161f2dd00634e0ef43fb",
        modelID: "qwen3.6-35b-a3b-6bit",
        approximateDownloadBytes: 29_081_792_392,
        installedBytes: 29_120_000_000,
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
    public static let `default` = gemma4

    public static let all: [SupportedModelSource] = [
        gemma4, qwen36, qwen36_6bit, qwen36_8bit, qwen36MTP,
    ]

    public static func named(_ name: String) -> SupportedModelSource? {
        all.first { $0.name == name }
    }
}
