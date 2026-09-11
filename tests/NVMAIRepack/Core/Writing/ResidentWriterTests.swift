import Foundation
import Testing
@testable import NVMAIRepackCore

/// The resident index stores each tensor name's length in a `UInt16`, and the
/// name comes from the source manifest. That makes an over-long name input
/// rather than an invariant: it has to be a report naming the tensor, not a
/// `precondition` trap inside the binary writer.
@Suite struct ResidentWriterTests {
    private func plan(name: String) -> ResidentFilePlan {
        let source = SourceTensor(name: name, shardPath: "shard.safetensors",
                                  dtype: .bf16, shape: [1], absoluteOffset: 0,
                                  sizeBytes: 2)
        let entry = ResidentEntry(name: name, dtype: 1, logicalShape4: [1, 0, 0, 0],
                                  fileOffset: 0, sizeBytes: 2,
                                  scaleOffset: 0, scaleSize: 0,
                                  biasOffset: 0, biasSize: 0,
                                  quantSpec: nil,
                                  sourceWeight: source,
                                  sourceScales: nil,
                                  sourceBiases: nil)
        let table = Array(name.utf8)
        return ResidentFilePlan(
            path: "/tmp/model_weights.bin",
            entries: [entry],
            stringTable: table,
            stringTableOffsets: [0],
            indexSize: UInt64(24 + GTurboBinary.indexEntryBytes + table.count),
            residentSize: 2)
    }

    @Test func aNameWithinTheLimitEncodes() throws {
        let data = try ResidentWriter.encodeIndex(plan: plan(name: "model.layers.0.mlp.up_proj.weight"))
        #expect(data.count == 24 + GTurboBinary.indexEntryBytes
                + "model.layers.0.mlp.up_proj.weight".utf8.count)
    }

    @Test func anOverLongNameIsReportedNotTrapped() {
        let name = String(repeating: "x", count: Int(UInt16.max) + 1)
        #expect(throws: RepackError.self) {
            _ = try ResidentWriter.encodeIndex(plan: plan(name: name))
        }
    }

    /// The boundary itself has to encode: a name of exactly `UInt16.max` bytes
    /// is what the field can hold, and refusing it would be a new limit.
    @Test func aNameOfExactlyTheLimitEncodes() throws {
        let name = String(repeating: "x", count: Int(UInt16.max))
        let data = try ResidentWriter.encodeIndex(plan: plan(name: name))
        #expect(data.count == 24 + GTurboBinary.indexEntryBytes + Int(UInt16.max))
    }
}
