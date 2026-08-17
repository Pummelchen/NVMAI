import Foundation
import NVMAI

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    /// The payload is complete and matches its manifest, but the install
    /// receipt was issued for a different directory — the model was moved or
    /// renamed. Re-attesting is a local re-hash, not a re-download, so this is
    /// deliberately not `.partial`: treating it as an incomplete install would
    /// offer the user a multi-gigabyte download for a model already on disk.
    case needsReattestation(String)
    case complete
}

public enum AppModelInstallationProbe {
    /// `descriptor` pins the checkpoint the installation must match. When it
    /// is nil the probe derives the expectation from the family the manifest
    /// itself declares, so a multi-model app validates whichever model is
    /// actually installed rather than whichever one is currently selected.
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor? = nil
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            let family = try ManifestReader.peekFamily(directoryURL: directory)
            guard let baseline = ArchConfig.knownArchitectures[family] else {
                return .partial("unknown model family \(family.rawValue)")
            }
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: baseline)
            // Validate the checkpoint against the descriptor for the family the
            // manifest itself declares, so the probe does not depend on which
            // model the app happens to have selected.
            let sourceMatched = manifest.sourceSnapshotHash.flatMap { hash in
                AppModelInstallDescriptor.all.first {
                    hash == "sha256:" + $0.sourceIndexSHA256
                }
            }
            guard let expected = descriptor ?? sourceMatched
                    ?? AppModelInstallDescriptor.descriptor(for: family) else {
                return .partial("no descriptor for family \(family.rawValue)")
            }
            let expectedSource = "sha256:" + expected.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(expected.displayName)")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            // Integrity first: a manifest mismatch means the payload is not
            // what was attested, and no local action fixes that.
            try VerifiedInstallReceiptReader.validateManifestIntegrity(
                receipt,
                manifestSha256: manifestHash)
            // Location second, and separately: a receipt bound to another path
            // means the directory moved, which a re-attestation fixes in place.
            guard VerifiedInstallReceiptReader.pathBindingMatches(
                receipt, directoryURL: directory) else {
                return .needsReattestation(
                    "the install receipt was issued for \(receipt.modelDirectoryPath)")
            }
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
