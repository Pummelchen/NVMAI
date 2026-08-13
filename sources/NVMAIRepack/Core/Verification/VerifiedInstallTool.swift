import Foundation
import Darwin
import NVMAIFormat

public struct VerifyInstallOptions: Sendable {
    public let inputGTurbo: String

    public init(inputGTurbo: String) {
        self.inputGTurbo = inputGTurbo
    }
}

public struct VerifyInstallResult: Sendable {
    public let receiptPath: String
    public let fileCount: Int
    public let bytesVerified: UInt64
    public let unexpectedEntries: [String]
}

public enum VerifiedInstallTool {
    // 64 MiB: sized for Qwen 3.6's ~22 MB layout.json (40 layers x 256 experts).
    public static let metadataMaxBytes: UInt64 = 64 * 1024 * 1024
    // Hard ceiling for any single payload file named by the manifest. The
    // manifest size remains the exact per-file contract (verified below); this
    // cap only rejects absurd corrupted manifests before hashing.
    public static let payloadMaxBytes: UInt64 = 64 * 1024 * 1024 * 1024

    public static func run(options: VerifyInstallOptions) throws -> VerifyInstallResult {
        let access = try GTurboDirectoryAccess(rootPath: options.inputGTurbo)
        let manifestPath = "manifest.json"
        try GTurboPathValidator.validateRelativePath(
            manifestPath, field: "manifest.files.manifest.json")
        let manifestSize = try access.fileSize(manifestPath)
        let manifestSha = try access.hash(manifestPath, noCache: true)
        let manifest = try loadManifest(access: access)
        try validatePackedExpertLayout(access: access, manifest: manifest)

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(manifest.files.count)
        var bytesVerified = manifestSize
        for relativePath in manifest.files.keys.sorted() {
            guard let entry = manifest.files[relativePath] else { continue }
            // Path validation before any filesystem operation: reject `..`,
            // absolute and non-normalized names, and duplicate filesystem keys.
            try GTurboPathValidator.validateRelativePath(
                relativePath, field: "manifest.files.\(relativePath)")
            guard entry.size <= payloadMaxBytes else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(entry.size) exceeds "
                        + "the \(payloadMaxBytes)-byte per-file cap")
            }
            // All reads go through the root-anchored openat chain with
            // O_NOFOLLOW at every level (no symlink escapes) and the
            // descriptor's type is verified after open.
            let actualSize = try access.fileSize(relativePath)
            guard actualSize == entry.size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != manifest \(entry.size)")
            }
            let actualSha = try access.hash(relativePath, noCache: true)
            guard actualSha.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relativePath) SHA mismatch")
            }
            let (verified, overflow) = bytesVerified.addingReportingOverflow(actualSize)
            guard !overflow else {
                throw RepackError.configurationInvalid(detail: "verified byte total overflows")
            }
            bytesVerified = verified
            files.append(RepackAudit.OutputFile(relativePath: relativePath,
                                                size: actualSize,
                                                sha256: actualSha))
        }
        let unexpectedEntries = try findUnexpectedEntries(access: access, manifest: manifest)

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: access.rootPath,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: nil,
            sourceRevision: manifest.sourceSnapshotHash,
            toolVersion: "NVMAIRepack verify-install",
            files: files)
        let receiptPath = access.rootPath
            + "/" + VerifiedInstallReceiptWriter.fileName
        try receiptData.write(to: URL(fileURLWithPath: receiptPath), options: .atomic)
        return VerifyInstallResult(receiptPath: receiptPath,
                                   fileCount: files.count + 1,
                                   bytesVerified: bytesVerified,
                                   unexpectedEntries: unexpectedEntries)
    }

    static func validatePackedExpertLayout(inputGTurbo: String) throws {
        let access = try GTurboDirectoryAccess(rootPath: inputGTurbo)
        let manifest = try loadManifest(access: access)
        try validatePackedExpertLayout(access: access, manifest: manifest)
    }

    private struct ManifestFileEntry: Decodable {
        let size: UInt64
        let sha256: String
    }

    private struct Manifest: Decodable {
        let files: [String: ManifestFileEntry]
        let expertsPerLayer: Int
        let numLayers: Int
        let expertStride: UInt64
        let sourceSnapshotHash: String?
    }

    private struct PackedExpertsLayout: Decodable {
        let expertStride: UInt64
        let numLayers: Int
        let expertsPerLayer: Int
        let layers: [Layer]
    }

    private struct Layer: Decodable {
        let layer: Int
        let file: String
        let experts: [Expert]
    }

    private struct Expert: Decodable {
        let expert: Int?
        let offset: UInt64
        let size: UInt64
    }

    private static func loadManifest(access: GTurboDirectoryAccess) throws -> Manifest {
        do {
            let data = try loadMetadataJSON(access: access, relativePath: "manifest.json")
            return try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private static func loadLayout(access: GTurboDirectoryAccess) throws -> PackedExpertsLayout {
        do {
            let data = try loadMetadataJSON(access: access,
                                            relativePath: "packed_experts/layout.json")
            return try JSONDecoder().decode(PackedExpertsLayout.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    private static func loadMetadataJSON(access: GTurboDirectoryAccess,
                                         relativePath: String) throws -> Data {
        try GTurboPathValidator.validateRelativePath(
            relativePath, field: "metadata.\(relativePath)")
        return try access.readMetadata(relativePath, maxBytes: metadataMaxBytes)
    }

    private static func validatePackedExpertLayout(access: GTurboDirectoryAccess,
                                                   manifest: Manifest) throws {
        let layoutRelativePath = "packed_experts/layout.json"
        guard manifest.files[layoutRelativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(layoutRelativePath)")
        }
        let layout = try loadLayout(access: access)
        let alignment = GTurboFormatV1.alignmentBytes
        guard layout.expertStride == manifest.expertStride,
              layout.numLayers == manifest.numLayers,
              layout.expertsPerLayer == manifest.expertsPerLayer else {
            throw RepackError.configurationInvalid(detail: "packed expert layout dimensions mismatch manifest")
        }
        guard layout.expertStride % alignment == 0 else {
            throw RepackError.configurationInvalid(
                detail: "expertStride \(layout.expertStride) is not aligned to \(alignment) bytes")
        }
        guard layout.layers.count == layout.numLayers else {
            throw RepackError.configurationInvalid(detail: "packed expert layout layer count mismatch")
        }
        let expectedLayerSize = UInt64(layout.expertsPerLayer) * layout.expertStride
        for layer in layout.layers {
            guard layer.layer >= 0 && layer.layer < layout.numLayers else {
                throw RepackError.configurationInvalid(detail: "packed expert layer index out of range")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw RepackError.configurationInvalid(
                    detail: "packed_experts/\(layer.file) expert count mismatch")
            }
            try GTurboPathValidator.validateBasename(
                layer.file, field: "packed_experts/layout.json layers[\(layer.layer)].file")
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
            }
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)")
            }
            let actualSize = try access.fileSize(relativePath)
            guard actualSize == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedLayerSize)")
            }
            var seenExperts = Set<Int>()
            for (index, expert) in layer.experts.enumerated() {
                let expertID = expert.expert ?? index
                guard expertID >= 0 && expertID < layout.expertsPerLayer else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert id out of range")
                }
                guard seenExperts.insert(expertID).inserted else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) duplicate expert \(expertID)")
                }
                guard expert.size == layout.expertStride else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) size mismatch")
                }
                guard expert.offset % GTurboFormatV1.alignmentBytes == 0 else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) offset is not aligned to \(GTurboFormatV1.alignmentBytes) bytes")
                }
                guard expert.offset <= actualSize,
                      expert.size <= actualSize - expert.offset else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
        }
    }

    private static func findUnexpectedEntries(access: GTurboDirectoryAccess,
                                              manifest: Manifest) throws -> [String] {
        let declaredFiles = Set(manifest.files.keys)
            .union(["manifest.json", VerifiedInstallReceiptWriter.fileName])
        var allowed = declaredFiles
        for path in declaredFiles {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                _ = parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }
        allowed.insert("tokenizer")

        let entries = try access.relativeEntries()
        var unexpected: [String] = []
        for rel in entries {
            // .DS_Store may appear at any depth (Finder writes it into
            // subdirectories too).
            if rel == ".DS_Store" || rel.hasSuffix("/.DS_Store") { continue }
            if rel == "tokenizer" || rel.hasPrefix("tokenizer/") { continue }
            if !allowed.contains(rel) {
                unexpected.append(rel)
            }
        }
        return unexpected.sorted()
    }
}
