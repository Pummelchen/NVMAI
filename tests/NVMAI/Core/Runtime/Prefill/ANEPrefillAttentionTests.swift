import Foundation
import Metal
import Testing
@testable import NVMAI

@Suite struct ANEPrefillAttentionTests {
    @Test func environmentSwitchDefaultsOffAndFailsClosed() throws {
        #expect(try RuntimePrefillANE.environmentValue([:]) == .off)
        #expect(try RuntimePrefillANE.environmentValue(
            ["NVMAI_PREFILL_ANE": "off"]) == .off)
        #expect(try RuntimePrefillANE.environmentValue(
            ["NVMAI_PREFILL_ANE": "on"]) == .on)
        #expect(throws: PrefillError.self) {
            try RuntimePrefillANE.environmentValue(["NVMAI_PREFILL_ANE": "1"])
        }
        #expect(throws: PrefillError.self) {
            try RuntimePrefillANE.environmentValue(["NVMAI_PREFILL_ANE": ""])
        }
    }

    @Test func missingSidecarFailsClosedWithExportHint() throws {
        let ctx = try MetalContext()
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(throws: PrefillError.self) {
            _ = try ANEPrefillAttention(modelDirectory: empty,
                                        device: ctx.device,
                                        hiddenSize: 2048, kvDim: 512,
                                        weightsSha256: nil)
        }
    }

    @Test func sidecarExportedFromDifferentWeightsIsRejected() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let meta: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 4096,
            "histories": [0], "layers": [3], "weightsSha256": String(repeating: "a", count: 64),
        ]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
        // Matching digest loads; a different one must fail closed rather than
        // computing plausible-looking attention from the wrong weights.
        _ = try ANEPrefillAttention(modelDirectory: dir, device: ctx.device,
                                    hiddenSize: 2048, kvDim: 512,
                                    weightsSha256: String(repeating: "A", count: 64))
        #expect(throws: PrefillError.self) {
            _ = try ANEPrefillAttention(modelDirectory: dir, device: ctx.device,
                                        hiddenSize: 2048, kvDim: 512,
                                        weightsSha256: String(repeating: "b", count: 64))
        }
    }

    /// Eligibility and shadow continuity, using a synthetic sidecar manifest
    /// so no Core ML package or model weights are involved.
    @Test func chunkEligibilityEnforcesAlignmentCoverageAndContinuity() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let meta: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 4096,
            "histories": [0, 4096], "layers": [3, 7],
        ]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
        let ane = try ANEPrefillAttention(modelDirectory: dir,
                                          device: ctx.device,
                                          hiddenSize: 2048, kvDim: 512,
                                          weightsSha256: nil)
        #expect(ane.maxPromptTokens == 8192)
        #expect(ane.coveredLayers == Set([3, 7]))

        // Config chunk mismatch, misaligned start, uncovered history: all out.
        #expect(!ane.eligibleChunk(startPosition: 0, tokenCount: 512,
                                   configChunkTokens: 1024))
        #expect(!ane.eligibleChunk(startPosition: 100, tokenCount: 4096,
                                   configChunkTokens: 4096))
        #expect(!ane.eligibleChunk(startPosition: 8192, tokenCount: 100,
                                   configChunkTokens: 4096))

        // A short single-chunk prompt stays on the GPU (padding waste).
        #expect(!ane.eligibleChunk(startPosition: 0, tokenCount: 512,
                                   configChunkTokens: 4096))
        // Fresh full-chunk prompt resets the shadow and is eligible.
        #expect(ane.eligibleChunk(startPosition: 0, tokenCount: 4096,
                                  configChunkTokens: 4096))
        // Without finishChunk, a follow-up chunk must fall back (continuity).
        #expect(!ane.eligibleChunk(startPosition: 4096, tokenCount: 100,
                                   configChunkTokens: 4096))
        ane.finishChunk(startPosition: 0, tokenCount: 4096)
        #expect(ane.shadowTokens == 4096)
        #expect(ane.eligibleChunk(startPosition: 4096, tokenCount: 100,
                                  configChunkTokens: 4096))
        // A partial final chunk clears the shadow: nothing may resume it.
        ane.finishChunk(startPosition: 4096, tokenCount: 100)
        #expect(ane.shadowTokens == 0)
    }

    @Test func shadowAppendSkipsPartialChunksAndStoresFullOnes() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let meta: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 8,
            "histories": [0, 8], "layers": [3],
        ]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
        let ane = try ANEPrefillAttention(modelDirectory: dir,
                                          device: ctx.device,
                                          hiddenSize: 16, kvDim: 4,
                                          weightsSha256: nil)
        let kPtr = ane.stagingK.contents().bindMemory(to: Float16.self,
                                                      capacity: 8 * 4)
        for index in 0..<(8 * 4) { kPtr[index] = Float16(index) }
        // Partial chunk: never appended (it is always the last chunk).
        ane.appendShadow(layer: 3, startPosition: 0, tokenCount: 4)
        ane.finishChunk(startPosition: 0, tokenCount: 4)
        #expect(ane.shadowTokens == 0)
        // Full chunk: appended and visible after finishChunk.
        ane.appendShadow(layer: 3, startPosition: 0, tokenCount: 8)
        ane.finishChunk(startPosition: 0, tokenCount: 8)
        #expect(ane.shadowTokens == 8)
    }
}
