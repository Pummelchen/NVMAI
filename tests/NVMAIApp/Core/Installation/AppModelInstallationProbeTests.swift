import Foundation
import Testing
@testable import NVMAIAppCore

@Suite struct AppModelInstallationProbeTests {
    @Test func missingDirectoryIsMissing() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("nvmai-missing-\(UUID().uuidString).gturbo")
        #expect(AppModelInstallationProbe.status(at: url) == .missing)
    }

    @Test func manifestWithoutFinalMetadataIsPartial() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("nvmai-partial-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{}".utf8).write(to: url.appendingPathComponent("manifest.json"))
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    @Test func validBoundedMetadataIsComplete() throws {
        let url = try makeCompleteModelInstall("probe")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
        #expect(AppModelInstallationProbe.status(
            at: url,
            descriptor: .ornith15) == .complete)
    }

    /// A receipt bound to another path used to report `.partial`, which the app
    /// reads as "not installed" and answers with an installer. It now reports
    /// `.needsReattestation`: the payload is intact and the fix is a local
    /// re-hash, not a multi-gigabyte download.
    @Test func receiptBoundToDifferentPathNeedsReattestation() throws {
        let url = try makeCompleteModelInstall("wrong-path")
        defer { try? FileManager.default.removeItem(at: url) }
        let receiptURL = url.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/different/model.gturbo"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        guard case .needsReattestation = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected needsReattestation status")
            return
        }
    }

    @Test func differentCheckpointIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-checkpoint")
        defer { try? FileManager.default.removeItem(at: url) }
        let descriptor = AppModelInstallDescriptor(
            displayName: "different",
            repoID: "example/different",
            revision: "revision",
            sourceIndexSHA256: String(repeating: "f", count: 64),
            approximateDownloadBytes: 1,
            installedBytes: 1,
            rangeStagingBytes: 1,
            reserveBytes: 1)
        guard case .partial = AppModelInstallationProbe.status(at: url, descriptor: descriptor) else {
            Issue.record("expected checkpoint mismatch to be partial")
            return
        }
    }
}

@Suite("Moved model installation")
struct MovedModelInstallationTests {
    /// Moving or renaming a complete model must not read as "not installed".
    ///
    /// The receipt binds to the absolute path it was issued for, so a move
    /// fails `validateManifestBinding`. The probe used to fold that into
    /// `.partial`, which drives `requiresModelInstallation` — so the app showed
    /// an installer, offering a multi-gigabyte re-download for a model already
    /// on disk. The payload is intact; only the attestation needs re-issuing.
    @Test func movedInstallNeedsReattestationNotReinstallation() throws {
        let original = try makeCompleteModelInstall("moved")
        defer { try? FileManager.default.removeItem(at: original) }
        #expect(AppModelInstallationProbe.status(at: original) == .complete)

        let moved = original.deletingLastPathComponent()
            .appendingPathComponent("nvmai-moved-\(UUID().uuidString).gturbo")
        try FileManager.default.moveItem(at: original, to: moved)
        defer { try? FileManager.default.removeItem(at: moved) }

        let status = AppModelInstallationProbe.status(at: moved)
        guard case .needsReattestation = status else {
            Issue.record("moved install reported \(status), expected .needsReattestation")
            return
        }
    }

    /// A payload that no longer matches its manifest is a different failure and
    /// must stay `.partial`: re-attesting cannot fix it, so the app must not
    /// offer that as the remedy.
    @Test func corruptedManifestStaysPartial() throws {
        let directory = try makeCompleteModelInstall("corrupt")
        defer { try? FileManager.default.removeItem(at: directory) }

        let manifestURL = directory.appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        manifest["numLayers"] = 999
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: manifestURL)

        let status = AppModelInstallationProbe.status(at: directory)
        if case .needsReattestation = status {
            Issue.record("a manifest mismatch must not be reported as re-attestable")
        }
    }
}
