import Foundation
import Metal
import Testing

@testable import NVMAI

extension ModelLoaderTests {
  @Test func loadsValidDirectory() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    let embed = try model.embedding()
    #expect(embed.length == UInt64(1024 * 64))
    #expect(embed.shape.0 == 1024 && embed.shape.1 == 64)
    let norm = try model.finalNorm()
    #expect(norm.length == UInt64(64 * 2))
    // Tied lm_head returns the same view as embedding.
    let lmHead = try model.lmHead()
    #expect(lmHead.offset == embed.offset)
    #expect(lmHead.length == embed.length)
  }

  @Test func residentBytesAreReadableFromBuffer() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    let norm = try model.finalNorm()
    let contents = norm.buffer.contents()
    // Norm region was patterned 0xC0 | (i & 0x3F).
    for i in 0..<Int(norm.length) {
      let got = contents.load(fromByteOffset: Int(norm.offset) + i, as: UInt8.self)
      #expect(got == UInt8(0xC0 | (i & 0x3F)), "norm byte \(i)")
    }
  }

  @Test func missingManifestFailsPartialInstall() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try FileManager.default.removeItem(at: dir.appendingPathComponent("manifest.json"))
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir, device: device,
        expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.partialInstall = error { return true }
      return false
    }
  }

  @Test func mismatchedShaFailsChecksumMismatch() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    // Flip one byte at the very end of the resident region (not inside
    // the index, which the loader reads earlier and would error
    // differently). Manifest sha was computed before this corruption.
    let url = dir.appendingPathComponent("model_weights.bin")
    var data = try Data(contentsOf: url)
    data[data.count - 1] ^= 0xFF
    try data.write(to: url)
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir, device: device,
        expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }
  }

  @Test func integrityPoliciesExposeIdenticalResidentAndRoutedBytes() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try #require(MTLCreateSystemDefaultDevice())
    let full = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .fullSha256)
    let trusted = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)

    let fullEmbedding = try full.embedding()
    let trustedEmbedding = try trusted.embedding()
    #expect(fullEmbedding.length == trustedEmbedding.length)
    let fullEmbeddingBytes = fullEmbedding.buffer.contents().advanced(by: Int(fullEmbedding.offset))
    let trustedEmbeddingBytes = trustedEmbedding.buffer.contents().advanced(
      by: Int(trustedEmbedding.offset))
    #expect(memcmp(fullEmbeddingBytes, trustedEmbeddingBytes, Int(fullEmbedding.length)) == 0)

    let fullExpert = try full.routedExpert(layer: 0, expert: 0)
    let trustedExpert = try trusted.routedExpert(layer: 0, expert: 0)
    #expect(fullExpert.length == trustedExpert.length)
    let fullExpertBytes = fullExpert.buffer.contents().advanced(by: Int(fullExpert.offset))
    let trustedExpertBytes = trustedExpert.buffer.contents().advanced(by: Int(trustedExpert.offset))
    #expect(memcmp(fullExpertBytes, trustedExpertBytes, Int(fullExpert.length)) == 0)
  }

  @Test func nonPageAlignedExpertStrideFailsAtManifest() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let manifestURL = dir.appendingPathComponent("manifest.json")
    var root =
      try JSONSerialization.jsonObject(
        with: Data(contentsOf: manifestURL)) as! [String: Any]
    root["expertStride"] = 1024
    let data = try JSONSerialization.data(withJSONObject: root)
    try data.write(to: manifestURL)
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir, device: device,
        expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.expertStrideNotPageAligned = error { return true }
      return false
    }
  }

}
