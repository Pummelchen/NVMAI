import Foundation
import Metal

/// Writes named activation slices to disk so a run can be compared against a
/// reference implementation stage by stage.
///
/// This exists because unit tests cannot tell a correct architecture from a
/// plausible one. Every mistake worth worrying about in a new family --- a
/// transposed gate, a norm applied to the wrong axis, a convolution tap read
/// one position off --- produces smooth, confident, wrong output. Dumping the
/// intermediates and diffing them against a reference is the only check that
/// localizes such a bug to a single stage.
///
/// Off unless `NVMAI_ACT_DUMP` names a directory, and every call site is
/// wrapped so a normal run does no work and takes no synchronization.
extension RealForwardRunner {
    /// Highest layer index the detailed per-layer dumps cover. Layers 0-3 span
    /// one full period of this family's attention schedule: three linear
    /// layers and the first full-attention one. The residual at every layer
    /// entry is dumped regardless -- it is 20 KB a layer and it is what
    /// locates the first stage where a run leaves the reference.
    var dumpLayerLimit: Int { 3 }

    /// Positions to dump, from 0. One is enough to check the per-layer math;
    /// more is what catches state threaded between tokens -- the KV cache,
    /// the delta-rule state, and the two convolution histories -- which a
    /// single position cannot exercise at all.
    static let dumpPositionCount = ProcessInfo.processInfo
        .environment["NVMAI_ACT_DUMP_POSITIONS"].flatMap(Int.init) ?? 1

    func activationDumpActive(position: Int) -> Bool {
        activationDumpDirectory != nil && position < Self.dumpPositionCount
    }

    /// Each position writes into its own subdirectory, so a later token does
    /// not quietly overwrite the one being compared.
    func activationDumpDirectory(position: Int) -> URL? {
        guard let root = activationDumpDirectory else { return nil }
        let directory = root.appendingPathComponent("pos\(position)")
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Records the token whose forward pass the dumps describe.
    func dumpActivationToken(_ token: Int32, position: Int) {
        guard let directory = activationDumpDirectory(position: position)
        else { return }
        try? "\(token)\n".write(
            to: directory.appendingPathComponent("token.txt"),
            atomically: true, encoding: .utf8)
    }

    /// Writes `count` fp16 values as raw little-endian fp16 (numpy
    /// `dtype='<f2'`). Synchronizes first: the caller passes the command
    /// buffer that produced the values, because reading a shared buffer the
    /// GPU is still writing would dump a mixture of two states and look like
    /// a numerical bug.
    func dumpActivation(_ name: String,
                        _ buffer: MTLBuffer,
                        count: Int,
                        position: Int,
                        offset: Int = 0,
                        after commandBuffer: MTLCommandBuffer? = nil) {
        guard let directory = activationDumpDirectory(position: position)
        else { return }
        commandBuffer?.waitUntilCompleted()
        let bytes = count * MemoryLayout<Float16>.stride
        guard offset + bytes <= buffer.length else { return }
        let data = Data(bytes: buffer.contents().advanced(by: offset),
                        count: bytes)
        try? data.write(to: directory.appendingPathComponent("\(name).f16"))
    }
}
