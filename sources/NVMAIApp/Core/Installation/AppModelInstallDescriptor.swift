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

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
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
        reserveBytes: 1_073_741_824)

    public static let qwen36_6bit = AppModelInstallDescriptor(
        displayName: "Qwen3.6 35B-A3B 6-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-6bit",
        revision: "cb7e092ef8efe540bc3672c8929c4adbe5f4f759",
        sourceIndexSHA256: "eaea194dfb961e6a5215dcc6e4dd42d0df6efe8d8686161f2dd00634e0ef43fb",
        approximateDownloadBytes: 29_081_792_392,
        installedBytes: 29_120_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    public static let qwen36_8bit = AppModelInstallDescriptor(
        displayName: "Qwen3.6 35B-A3B 8-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-8bit",
        revision: "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
        sourceIndexSHA256: "3db12edeebeb65cab9a6eeb63cd74be4e0c74139a75f672701290b98230501cf",
        approximateDownloadBytes: 37_741_392_345,
        installedBytes: 37_800_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    public static let ornith15 = AppModelInstallDescriptor(
        displayName: "Ornith 1.5 35B-A3B 4-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        revision: "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
        sourceIndexSHA256: "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36",
        approximateDownloadBytes: 19_528_995_943,
        installedBytes: 19_530_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    public static let ornith15_8bit = AppModelInstallDescriptor(
        displayName: "Ornith 1.5 35B-A3B 8-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
        revision: "02440c39bdf7365c494a7f55f2a8b104ba87562f",
        sourceIndexSHA256: "83c641a791aa957df7d280eef1b0c8faf7a2ec9b19dd3355fb13abae8ae0ed15",
        approximateDownloadBytes: 36_848_194_663,
        installedBytes: 36_850_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    /// Known source fingerprints, including withdrawn 6-bit installations so
    /// an existing receipt can still be identified and reported accurately.
    public static let all: [AppModelInstallDescriptor] = [
        .qwen36, .qwen36_6bit, .qwen36_8bit, .ornith15, .ornith15_8bit,
    ]

    /// The shipped descriptor for a model family, if one exists.
    public static func descriptor(for family: ModelFamily) -> AppModelInstallDescriptor? {
        switch family {
        case .qwen36: return .qwen36
        case .qwen36MTP: return nil
        }
    }

    /// Basename of the installed `.gturbo` directory for this descriptor.
    public var installDirectoryName: String {
        switch repoID {
        case Self.ornith15_8bit.repoID: return "ornith-1.5_35B_A3B_8Bit"
        case Self.ornith15.repoID: return "ornith-1.5_35B_A3B_4Bit"
        case Self.qwen36_6bit.repoID: return "qwen3.6_35B_A3B_6Bit"
        case Self.qwen36_8bit.repoID: return "qwen3.6_35B_A3B_8Bit"
        case Self.qwen36.repoID: return "qwen3.6_35B_A3B_4Bit"
        default: return "ornith-1.5_35B_A3B_4Bit"
        }
    }

    /// The descriptor the app products select at launch. Defaults to Ornith 1.5
    /// 4-bit. `TURBO_FIELDFARE_MODEL` in the environment wins; otherwise the
    /// persisted `defaults write NVMAI model <selector>` preference applies.
    public static var selected: AppModelInstallDescriptor {
        let environmentValue = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MODEL"]
        let preferenceValue = UserDefaults(suiteName: "NVMAI")?
            .string(forKey: "model")
        return selectedDescriptor(for: environmentValue ?? preferenceValue)
    }

    static func selectedDescriptor(for selector: String?) -> AppModelInstallDescriptor {
        switch selector {
        case "qwen36": return .qwen36
        case "qwen36-8bit": return .qwen36_8bit
        case "ornith15": return .ornith15
        case "ornith15-8bit": return .ornith15_8bit
        default: return .ornith15
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
