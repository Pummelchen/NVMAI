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

    public static let all: [AppModelInstallDescriptor] = [
        .qwen36, .qwen36_6bit, .qwen36_8bit,
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
        case Self.qwen36_6bit.repoID: return "qwen3.6_35B_A3B_6Bit"
        case Self.qwen36_8bit.repoID: return "qwen3.6_35B_A3B_8Bit"
        case Self.qwen36.repoID: return "qwen3.6_35B_A3B_4Bit"
        default: return "qwen3.6_35B_A3B_4Bit"
        }
    }

    /// The descriptor the app products select at launch. Defaults to Qwen 3.6
    /// 4-bit. `TURBO_FIELDFARE_MODEL=qwen36` in the environment wins;
    /// otherwise the persisted preference (`defaults write NVMAI model
    /// qwen36`) applies, so GUI launches without an environment also select
    /// Qwen.
    public static var selected: AppModelInstallDescriptor {
        let environmentValue = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MODEL"]
        let preferenceValue = UserDefaults(suiteName: "NVMAI")?
            .string(forKey: "model")
        switch environmentValue ?? preferenceValue {
        case "qwen36": return .qwen36
        case "qwen36-6bit": return .qwen36_6bit
        case "qwen36-8bit": return .qwen36_8bit
        default: return .qwen36
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
