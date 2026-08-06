import Foundation

public enum ModelIntegrityPolicy: Sendable, Equatable {
    case fullSha256
    case sizeCheckTrustedReceipt

    /// Strongest policy that skips re-hashing the full payload: when the
    /// installer's `verified-install.json` receipt is present, the per-file
    /// hashes were already pinned at install time, so loads verify the
    /// receipt binding and file sizes instead of re-hashing the weights.
    /// Without a receipt, fall back to a full SHA-256 of everything.
    public static func resolved(directoryURL: URL) -> ModelIntegrityPolicy {
        let receiptURL = directoryURL
            .appendingPathComponent(VerifiedInstallReceiptReader.fileName)
        return FileManager.default.fileExists(atPath: receiptURL.path)
            ? .sizeCheckTrustedReceipt
            : .fullSha256
    }
}

public struct VerifiedInstallReceipt: Codable, Equatable, Sendable {
    public struct FileEntry: Codable, Equatable, Sendable {
        public let size: UInt64
        public let sha256: String

        public init(size: UInt64, sha256: String) {
            self.size = size
            self.sha256 = sha256
        }
    }

    public let schemaVersion: Int
    public let manifestSha256: String
    public let modelDirectoryPath: String
    public let sourceRepoID: String?
    public let sourceRevision: String?
    public let verificationTimestamp: String
    public let toolVersion: String
    public let files: [String: FileEntry]

    public init(schemaVersion: Int = 1,
                manifestSha256: String,
                modelDirectoryPath: String,
                sourceRepoID: String? = nil,
                sourceRevision: String? = nil,
                verificationTimestamp: String,
                toolVersion: String,
                files: [String: FileEntry]) {
        self.schemaVersion = schemaVersion
        self.manifestSha256 = manifestSha256
        self.modelDirectoryPath = modelDirectoryPath
        self.sourceRepoID = sourceRepoID
        self.sourceRevision = sourceRevision
        self.verificationTimestamp = verificationTimestamp
        self.toolVersion = toolVersion
        self.files = files
    }
}

public enum VerifiedInstallReceiptReader {
    public static let fileName = "verified-install.json"
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    public static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> VerifiedInstallReceipt {
        let url = directoryURL.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName) is missing")
        }
        do {
            // K17: read first, then verify the size against the same data —
            // checking attributesOfItem and then re-reading the file was a
            // TOCTOU window (the file could grow between the two).
            let data = try Data(contentsOf: url)
            guard UInt64(data.count) <= maxBytes else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(fileName) size \(data.count) exceeds metadata cap \(maxBytes)")
            }
            return try JSONDecoder().decode(VerifiedInstallReceipt.self, from: data)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        }
    }

    public static func validate(_ receipt: VerifiedInstallReceipt,
                                directoryURL: URL,
                                manifest: Manifest,
                                manifestSha256: String,
                                manifestSize: UInt64) throws {
        try validateManifestBinding(receipt,
                                    directoryURL: directoryURL,
                                    manifestSha256: manifestSha256)
        var expectedFiles = Set(manifest.files.keys)
        expectedFiles.insert("manifest.json")
        let receiptFiles = Set(receipt.files.keys)
        guard receiptFiles == expectedFiles else {
            let missing = expectedFiles.subtracting(receiptFiles).sorted()
            let extra = receiptFiles.subtracting(expectedFiles).sorted()
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt file set mismatch missing=\(missing) extra=\(extra)")
        }
        guard let manifestReceiptEntry = receipt.files["manifest.json"] else {
            throw ModelError.trustedReceiptInvalid(detail: "receipt missing manifest.json")
        }
        guard manifestReceiptEntry.size == manifestSize else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json size mismatch")
        }
        guard manifestReceiptEntry.sha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json SHA mismatch")
        }

        for (rel, manifestEntry) in manifest.files {
            guard let receiptEntry = receipt.files[rel] else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt missing \(rel)")
            }
            guard receiptEntry.size == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt size mismatch for \(rel)")
            }
            guard receiptEntry.sha256.lowercased() == manifestEntry.sha256.lowercased() else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt SHA mismatch for \(rel)")
            }
        }
    }

    public static func validateManifestBinding(_ receipt: VerifiedInstallReceipt,
                                               directoryURL: URL,
                                               manifestSha256: String) throws {
        guard receipt.schemaVersion == 1 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "unsupported schemaVersion \(receipt.schemaVersion)")
        }
        guard receipt.manifestSha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest SHA mismatch")
        }

        let actualPath = directoryURL.standardizedFileURL.path
        guard receipt.modelDirectoryPath == actualPath else {
            throw ModelError.trustedReceiptInvalid(detail: "model directory mismatch")
        }
    }
}
