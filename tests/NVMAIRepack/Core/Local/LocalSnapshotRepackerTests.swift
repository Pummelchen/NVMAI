import Foundation
import Testing
@testable import NVMAIRepackCore

@Suite struct LocalSnapshotRepackerTests {
    @Test func importsMTPWithExplicitModelIdentityAndReceipt() async throws {
        let root = temporaryRoot("local-mtp-import")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)

        let result = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "ornith-1.5-35b-a3b-mtp-4bit",
                minFreeReserveBytes: 0))

        #expect(result.outputDir == output)
        #expect(result.rangeRequestCount == 0)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifestObject = try JSONSerialization.jsonObject(with: manifestData)
        let manifest = try #require(manifestObject as? [String: Any])
        #expect(manifest["modelID"] as? String
            == "ornith-1.5-35b-a3b-mtp-4bit")
        #expect((manifest["arch"] as? [String: Any])?["family"] as? String
            == "qwen36_mtp")
        #expect(try Posix.entryKind((output as NSString)
            .appendingPathComponent("verified-install.json")) == .regular)
        #expect(try Posix.entryKind((output as NSString)
            .appendingPathComponent("packed_experts/layer_00.bin")) == .regular)
    }

    @Test func rejectsUnsafeShardPathBeforeCopying() throws {
        let root = temporaryRoot("local-mtp-unsafe")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)
        let indexPath = (snapshot as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        let indexObject = try JSONSerialization.jsonObject(with: data)
        var index = try #require(indexObject as? [String: Any])
        var weightMap = try #require(index["weight_map"] as? [String: String])
        for name in weightMap.keys { weightMap[name] = "../outside.safetensors" }
        index["weight_map"] = weightMap
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: indexPath))

        #expect(throws: RepackError.self) {
            _ = try LocalSnapshotLoader.load(directory: snapshot)
        }
    }

    private func temporaryRoot(_ tag: String) -> String {
        let base = (FileManager.default.currentDirectoryPath as NSString)
            .appendingPathComponent(".build/test-artifacts")
        try? FileManager.default.createDirectory(
            atPath: base, withIntermediateDirectories: true)
        let path = (base as NSString)
            .appendingPathComponent("\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: path,
                                                 withIntermediateDirectories: true)
        return path
    }
}
