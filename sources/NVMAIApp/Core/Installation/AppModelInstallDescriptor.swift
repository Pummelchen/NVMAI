import Foundation
import NVMAI
import NVMAIRepackCore

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64
    /// Basename of the installed directory this checkpoint lands in. Stored,
    /// not derived from `repoID`: the bf16-sourced builds take both widths
    /// from one repository, so the repository cannot name the directory.
    public let installDirectoryName: String
    /// Whether the app's own downloader can produce this build. The app
    /// repacks an MLX-format checkpoint; the bf16-sourced builds are made by
    /// `tools/install_models.sh`, which runs a Python converter the app does
    /// not carry. Those are recognized and run, never offered as a download.
    public let isInstallable: Bool

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64,
                installDirectoryName: String,
                isInstallable: Bool = true) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
        self.installDirectoryName = installDirectoryName
        self.isInstallable = isInstallable
    }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    public static let qwen36 = AppModelInstallDescriptor(
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256: "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        approximateDownloadBytes: 19_529_025_048,
        installedBytes: 19_546_491_213,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        installDirectoryName: "qwen3.6_35B_A3B_4Bit")

    public static let qwen36_6bit = AppModelInstallDescriptor(
        displayName: "Qwen3.6 35B-A3B 6-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-6bit",
        revision: "cb7e092ef8efe540bc3672c8929c4adbe5f4f759",
        sourceIndexSHA256: "eaea194dfb961e6a5215dcc6e4dd42d0df6efe8d8686161f2dd00634e0ef43fb",
        approximateDownloadBytes: 29_081_792_392,
        installedBytes: 29_120_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        installDirectoryName: "qwen3.6_35B_A3B_6Bit")

    public static let qwen36_8bit = AppModelInstallDescriptor(
        displayName: "Qwen3.6 35B-A3B 8-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-8bit",
        revision: "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
        sourceIndexSHA256: "3db12edeebeb65cab9a6eeb63cd74be4e0c74139a75f672701290b98230501cf",
        approximateDownloadBytes: 37_741_392_345,
        installedBytes: 37_800_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        installDirectoryName: "qwen3.6_35B_A3B_8Bit")

    public static let ornith15 = AppModelInstallDescriptor(
        displayName: "Ornith 1.5 35B-A3B 4-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        revision: "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
        sourceIndexSHA256: "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36",
        approximateDownloadBytes: 19_528_995_943,
        installedBytes: 19_530_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        installDirectoryName: "ornith-1.5_35B_A3B_4Bit")

    public static let ornith15_8bit = AppModelInstallDescriptor(
        displayName: "Ornith 1.5 35B-A3B 8-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
        revision: "02440c39bdf7365c494a7f55f2a8b104ba87562f",
        sourceIndexSHA256: "83c641a791aa957df7d280eef1b0c8faf7a2ec9b19dd3355fb13abae8ae0ed15",
        approximateDownloadBytes: 36_848_194_663,
        installedBytes: 36_850_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        installDirectoryName: "ornith-1.5_35B_A3B_8Bit")


    // The builds `tools/install_models.sh` produces: quantized from each
    // model's own bf16 release, embedding and head at 8-bit. Recognized and
    // run by the app, installed from the command line (see `isInstallable`).
    private static func converted(_ name: String, _ repo: String, _ hash: String,
                                  _ installed: UInt64, _ directory: String)
        -> AppModelInstallDescriptor {
        AppModelInstallDescriptor(
            displayName: name, repoID: repo, revision: "converted",
            sourceIndexSHA256: hash,
            approximateDownloadBytes: installed, installedBytes: installed,
            rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
            reserveBytes: 1_073_741_824,
            installDirectoryName: directory, isInstallable: false)
    }

    public static let ornith15Converted = converted(
        "Ornith 1.5 35B-A3B 4-bit", "ornith-ai/Ornith-1.5-35B-A3B",
        "84384b8c50bc24538a561491f95974fb3289113fb890ba5ad4cecc6d7a5d8a02",
        19_530_000_000, "ornith-1.5_35B_A3B_4Bit")
    public static let ornith15Converted8bit = converted(
        "Ornith 1.5 35B-A3B 8-bit", "ornith-ai/Ornith-1.5-35B-A3B",
        "e37fb5771c9105395b0948fe3ef2351ab2e883576756d83c1cb76a5d2635db90",
        36_850_000_000, "ornith-1.5_35B_A3B_8Bit")
    public static let qwen36Converted = converted(
        "Qwen 3.6 35B-A3B 4-bit", "Qwen/Qwen3.6-35B-A3B",
        "9d367d912c25c3568f1d252d5f349fc5c774d59ced93574c31fb6d6dd4f42fb8",
        19_546_000_000, "qwen3.6_35B_A3B_4Bit")
    public static let qwen36Converted8bit = converted(
        "Qwen 3.6 35B-A3B 8-bit", "Qwen/Qwen3.6-35B-A3B",
        "727d4428f484683ff775415c359dfa0c5acd5d83b0a12b7ad03a7dc7de54ff87",
        37_800_000_000, "qwen3.6_35B_A3B_8Bit")
    public static let agentworld = converted(
        "Qwen-AgentWorld 35B-A3B 4-bit", "Qwen/Qwen-AgentWorld-35B-A3B",
        "25f44babb4feb9d9070b1cba1d5e5d70c41e70d82f5438a13f6d55e112808d05",
        19_546_000_000, "qwen-agentworld_35B_A3B_4Bit")
    public static let agentworld8bit = converted(
        "Qwen-AgentWorld 35B-A3B 8-bit", "Qwen/Qwen-AgentWorld-35B-A3B",
        "cf0056887d0985c96aae3939a8bdcd371a1c762dac49ea9bd07e549e42108b95",
        37_800_000_000, "qwen-agentworld_35B_A3B_8Bit")
    public static let qwen38 = converted(
        "Qwen3.8-Flash-Next 125B-A6B 4-bit", "Qwen/Qwen3.8-Flash-Next",
        "331102fda39f492e5957d4f773d78927addc6b998570e52389c748c84966be23",
        161_000_000_000, "qwen3.8-flash-next_125B_A6B_4Bit")
    public static let qwen38_8bit = converted(
        "Qwen3.8-Flash-Next 125B-A6B 8-bit", "Qwen/Qwen3.8-Flash-Next",
        "ea2295ffa131200158083b15463b8d2f4f9c8b4b1996972a456b6281a29befa2",
        197_000_000_000, "qwen3.8-flash-next_125B_A6B_8Bit")

    /// Every source fingerprint the app recognizes: the builds the installer
    /// produces today, plus the MLX repacks the app can still download and
    /// the withdrawn 6-bit one, so an existing receipt is always identified
    /// rather than reported as a foreign checkpoint.
    public static let all: [AppModelInstallDescriptor] = [
        .ornith15Converted, .ornith15Converted8bit,
        .qwen36Converted, .qwen36Converted8bit,
        .agentworld, .agentworld8bit,
        .qwen38, .qwen38_8bit,
        .qwen36, .qwen36_6bit, .qwen36_8bit, .ornith15, .ornith15_8bit,
    ]

    /// The `tools/install_models.sh` target that produces this build, for the
    /// instruction the app shows when it cannot download one itself.
    public var installerTarget: String {
        switch installDirectoryName {
        case "ornith-1.5_35B_A3B_4Bit": return "ornith15"
        case "ornith-1.5_35B_A3B_8Bit": return "ornith15-8bit"
        case "qwen3.6_35B_A3B_4Bit": return "qwen36"
        case "qwen3.6_35B_A3B_8Bit": return "qwen36-8bit"
        case "qwen-agentworld_35B_A3B_4Bit": return "agentworld"
        case "qwen-agentworld_35B_A3B_8Bit": return "agentworld-8bit"
        case "qwen3.8-flash-next_125B_A6B_4Bit": return "qwen38flash"
        case "qwen3.8-flash-next_125B_A6B_8Bit": return "qwen38flash-8bit"
        default: return "--help"
        }
    }

    /// The subset the app can download and repack itself.
    public static var installable: [AppModelInstallDescriptor] {
        all.filter(\.isInstallable)
    }

    /// The shipped descriptor for a model family, if one exists.
    public static func descriptor(for family: ModelFamily) -> AppModelInstallDescriptor? {
        switch family {
        case .qwen36: return .qwen36Converted
        case .qwen36MTP: return nil
        // Qwen3.8-Flash-Next is recognized so an existing install runs, but
        // it is never an app download: the model and its 102 GB n-gram table
        // are a CLI-scale install, and the draft head is only meaningful
        // beside a target that is already there.
        case .qwen38flash: return .qwen38
        case .qwen38flashMTP: return nil
        }
    }

    /// The descriptor the app products select at launch. Defaults to Ornith 1.5
    /// 8-bit. `TURBO_FIELDFARE_MODEL` in the environment wins; otherwise the
    /// persisted `defaults write NVMAI model <selector>` preference applies.
    public static var selected: AppModelInstallDescriptor {
        let environmentValue = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MODEL"]
        let preferenceValue = UserDefaults(suiteName: "NVMAI")?
            .string(forKey: "model")
        return selectedDescriptor(for: environmentValue ?? preferenceValue)
    }

    static func selectedDescriptor(for selector: String?) -> AppModelInstallDescriptor {
        switch selector {
        // The eight installs `tools/install_models.sh` produces.
        case "ornith15", "ornith": return .ornith15Converted
        case "ornith15-8bit", "ornith-8bit": return .ornith15Converted8bit
        case "qwen36", "qwen3.6": return .qwen36Converted
        case "qwen36-8bit", "qwen3.6-8bit": return .qwen36Converted8bit
        case "agentworld": return .agentworld
        case "agentworld-8bit": return .agentworld8bit
        case "qwen38", "qwen3.8": return .qwen38
        case "qwen38-8bit", "qwen3.8-8bit": return .qwen38_8bit
        // The MLX repacks the app can still download.
        case "ornith15-mlx": return .ornith15
        case "ornith15-8bit-mlx": return .ornith15_8bit
        case "qwen36-mlx": return .qwen36
        case "qwen36-8bit-mlx": return .qwen36_8bit
        case "qwen36-6bit": return .qwen36_6bit
        default: return .ornith15Converted8bit
        }
    }
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .checking, .failed:
            return nil
        }
    }
}
