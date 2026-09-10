import Foundation
import Testing
@testable import NVMAI

/// The side-engine's loader and its scalar arithmetic.
///
/// The whole-model check lives in `NVMAIBench cpu35`, because it needs a two
/// gigabyte snapshot and agreement with `tools/qwen35_reference.py` — both
/// measured, both recorded in `docs/plan-cpu-side-engine.md`. What is here is
/// everything that can be checked without one, which is the loader's contract
/// and the arithmetic between the GEMVs.
@Suite struct CPUEngineTests {

    // MARK: - safetensors

    /// Builds a real safetensors file: an eight-byte header length, a JSON
    /// header, then the payload. Written by hand rather than by a library so
    /// the test fails if the reader's idea of the format drifts.
    private func writeShard(_ tensors: [(name: String, dtype: String,
                                         shape: [Int], bytes: [UInt8])]) throws -> URL {
        var header: [String: Any] = [:]
        var payload: [UInt8] = []
        for tensor in tensors {
            header[tensor.name] = ["dtype": tensor.dtype, "shape": tensor.shape,
                                   "data_offsets": [payload.count,
                                                    payload.count + tensor.bytes.count]]
            payload.append(contentsOf: tensor.bytes)
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        // The payload has to start on the offset the header length declares;
        // safetensors pads the header with spaces to whatever alignment the
        // writer likes, and the reader must honour the declared length.
        while json.count % 8 != 0 { json.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(contentsOf: payload)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("cpu-engine-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        return url
    }

    private func bf16(_ values: [Float]) -> [UInt8] {
        var bytes: [UInt8] = []
        for value in values {
            let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
            withUnsafeBytes(of: bits.littleEndian) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    @Test func readsShapesAndOffsets() throws {
        let url = try writeShard([
            ("a", "F32", [2, 3], [UInt8](repeating: 0, count: 24)),
            ("b", "BF16", [4], bf16([1, 2, 3, 4])),
        ])
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafeTensorsFile(url: url)
        #expect(file.entries.count == 2)
        #expect(try file.entry("a").shape == [2, 3])
        #expect(try file.entry("b").dtype == "BF16")
    }

    /// BF16 is the top sixteen bits of a float32, and the reader widens by
    /// bit pattern rather than by a library — because the one it would have
    /// used cannot decode BF16 at all, which is how this reader came to
    /// exist.
    @Test func widensBFloatByBitPattern() throws {
        let values: [Float] = [1, -2, 0.5, 1024, 0]
        let url = try writeShard([("g", "BF16", [values.count], bf16(values))])
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafeTensorsFile(url: url)
        #expect(try file.floats("g") == values, "these all round-trip exactly")
    }

    @Test func missingTensorIsAnError() throws {
        let url = try writeShard([("a", "F32", [1], [0, 0, 0, 0])])
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafeTensorsFile(url: url)
        #expect(throws: SafeTensorsFile.Failure.self) { try file.bytes("absent") }
    }

    // MARK: - a whole (tiny) snapshot

    /// A snapshot with one quantized matrix, written the way the converter
    /// writes one: `bits`-wide unsigned lanes packed low-first into UInt32,
    /// one BF16 scale and bias per group.
    private func writeSnapshot(rows: Int, columns: Int, bits: Int,
                               group: Int = 64,
                               level: (Int, Int) -> UInt32,
                               scale: Float, bias: Float,
                               overrides: [String: Int] = [:]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("snapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let lanes = 32 / bits
        var words: [UInt32] = []
        for row in 0..<rows {
            for word in 0..<(columns / lanes) {
                var packed: UInt32 = 0
                for lane in 0..<lanes {
                    packed |= (level(row, word * lanes + lane) & UInt32((1 << bits) - 1))
                        << (bits * lane)
                }
                words.append(packed)
            }
        }
        var weightBytes: [UInt8] = []
        for word in words {
            withUnsafeBytes(of: word.littleEndian) { weightBytes.append(contentsOf: $0) }
        }
        let groups = rows * (columns / group)
        let url = try writeShard([
            ("w.weight", "U32", [rows, columns / lanes], weightBytes),
            ("w.scales", "BF16", [rows, columns / group],
             bf16([Float](repeating: scale, count: groups))),
            ("w.biases", "BF16", [rows, columns / group],
             bf16([Float](repeating: bias, count: groups))),
        ])
        try FileManager.default.moveItem(
            at: url, to: directory.appendingPathComponent("model.safetensors"))

        var quantization: [String: Any] = ["bits": bits, "group_size": group, "mode": "affine"]
        for (stem, width) in overrides {
            quantization[stem] = ["bits": width, "group_size": group]
        }
        let config: [String: Any] = [
            "hidden_size": columns, "num_hidden_layers": 1, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": columns, "full_attention_interval": 4,
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": columns, "linear_value_head_dim": columns,
            "linear_conv_kernel_dim": 4, "intermediate_size": columns,
            "vocab_size": rows, "rms_norm_eps": 1e-6, "quantization": quantization,
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: [
            "weight_map": ["w.weight": "model.safetensors",
                           "w.scales": "model.safetensors",
                           "w.biases": "model.safetensors"]])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    @Test func readsAQuantizedMatrixAtBothWidths() throws {
        for bits in [4, 8] {
            let directory = try writeSnapshot(rows: 128, columns: 64, bits: bits,
                                              level: { _, _ in 1 }, scale: 1, bias: 0)
            defer { try? FileManager.default.removeItem(at: directory) }
            let snapshot = try AffineSnapshot(directory: directory)
            let matrix = try snapshot.matrix("w.weight")
            #expect(matrix.rows == 128)
            #expect(matrix.columns == 64,
                    Comment(rawValue: "columns must be unpacked, not the stored word count"))
            #expect(matrix.bits == bits)
        }
    }

    /// The 4-bit build keeps some tensors at 8 bits, so the width is per
    /// tensor and the base is only a default. Reading it as a snapshot-wide
    /// constant is a bug this project has already had once.
    @Test func perTensorWidthOverridesTheBase() throws {
        let directory = try writeSnapshot(rows: 128, columns: 64, bits: 8,
                                          level: { _, _ in 1 }, scale: 1, bias: 0,
                                          overrides: ["w": 8])
        defer { try? FileManager.default.removeItem(at: directory) }
        let snapshot = try AffineSnapshot(directory: directory)
        #expect(snapshot.bits(forStem: "w") == 8)
        #expect(snapshot.bits(forStem: "elsewhere") == 8)
    }

    /// `y = W · x` where every level is 1 and the scale is 1: each row sums
    /// x. A wrong lane order or group stride shows up immediately.
    @Test func gemvComputesTheProduct() throws {
        for bits in [4, 8] {
            let columns = 128, rows = 256
            let directory = try writeSnapshot(rows: rows, columns: columns, bits: bits,
                                              level: { _, column in UInt32(column % 2) },
                                              scale: 1, bias: 0)
            defer { try? FileManager.default.removeItem(at: directory) }
            let snapshot = try AffineSnapshot(directory: directory)
            let matrix = try snapshot.matrix("w.weight")
            let x = (0..<columns).map { Float($0) }
            var out = [Float](repeating: 0, count: rows)
            x.withUnsafeBufferPointer { input in
                out.withUnsafeMutableBufferPointer { output in
                    CPUOps.gemv(matrix, x: input.baseAddress!,
                                out: output.baseAddress!, threads: 4)
                }
            }
            // Levels alternate 0,1 by column, so each row sums the odd x.
            let expected = stride(from: 1, to: columns, by: 2).reduce(Float(0)) { $0 + Float($1) }
            #expect(out.allSatisfy { abs($0 - expected) < 0.01 },
                    Comment(rawValue: "\(bits)-bit: got \(out[0]), wanted \(expected)"))
        }
    }

    /// Threading splits rows, so it cannot change the answer. The kernel's
    /// own tests assert this for INT8; this asserts it through the path the
    /// engine actually uses, including INT4.
    @Test func threadingDoesNotChangeTheResult() throws {
        let columns = 128, rows = 512
        for bits in [4, 8] {
            let directory = try writeSnapshot(
                rows: rows, columns: columns, bits: bits,
                level: { row, column in UInt32((row &+ column) % (1 << bits)) },
                scale: 0.01, bias: -0.5)
            defer { try? FileManager.default.removeItem(at: directory) }
            let snapshot = try AffineSnapshot(directory: directory)
            let matrix = try snapshot.matrix("w.weight")
            let x = (0..<columns).map { Float($0 % 7) * 0.25 }
            func run(_ threads: Int) -> [Float] {
                var out = [Float](repeating: 0, count: rows)
                x.withUnsafeBufferPointer { input in
                    out.withUnsafeMutableBufferPointer { output in
                        CPUOps.gemv(matrix, x: input.baseAddress!,
                                    out: output.baseAddress!, threads: threads)
                    }
                }
                return out
            }
            #expect(run(1) == run(4), Comment(rawValue: "\(bits)-bit threading must be exact"))
            #expect(run(4) == run(8))
        }
    }

    // MARK: - the width policy

    /// A snapshot with no layers at all: enough for `CPUQwen35` to
    /// initialize, which is all the width policy needs, and it exercises the
    /// real initializer rather than a stand-in.
    private func writeEmptyModel(hidden: Int = 64) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        let url = try writeShard([
            ("language_model.model.norm.weight", "BF16", [hidden],
             bf16([Float](repeating: 1, count: hidden))),
        ])
        try FileManager.default.moveItem(
            at: url, to: directory.appendingPathComponent("model.safetensors"))
        let config: [String: Any] = [
            "hidden_size": hidden, "num_hidden_layers": 0, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": hidden, "full_attention_interval": 4,
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": hidden, "linear_value_head_dim": hidden,
            "linear_conv_kernel_dim": 4, "intermediate_size": hidden,
            "vocab_size": 8, "rms_norm_eps": 1e-6,
            "quantization": ["bits": 8, "group_size": 64, "mode": "affine"],
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: [
            "weight_map": ["language_model.model.norm.weight": "model.safetensors"]])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// Width is a scheduling decision, and the whole policy is one line:
    /// narrow while a person is waiting on the main engine, wide in the
    /// gaps. Measured, four threads costs a 35B generation 31% and one
    /// costs 3%, so this is the difference between a side-engine and a
    /// tax on the answer someone asked for.
    ///
    /// Exercised through a tiny snapshot rather than the real model, because
    /// what is being tested is the decision, not the arithmetic.
    @Test func widthFollowsContention() throws {
        let directory = try writeEmptyModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try CPUQwen35(snapshot: try AffineSnapshot(directory: directory),
                                  threads: 4)
        #expect(model.threads == 4, "without a signal the width is whatever was asked for")

        let busy = Busy()
        model.contention = { busy.value }
        model.busyThreads = 1
        model.idleThreads = 4

        busy.value = true
        model.applyWidthPolicy()
        #expect(model.threads == 1, "a person is waiting; take one core")

        busy.value = false
        model.applyWidthPolicy()
        #expect(model.threads == 4, "nothing waiting; take the performance cores")
    }

    /// A signal that never fires leaves the width exactly as configured, so
    /// a caller that does not care about contention is not surprised by it.
    @Test func noSignalLeavesTheWidthAlone() throws {
        let directory = try writeEmptyModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try CPUQwen35(snapshot: try AffineSnapshot(directory: directory),
                                  threads: 3)
        model.applyWidthPolicy()
        #expect(model.threads == 3)
    }

    /// unchecked-invariant: written and read only from the test's own thread,
    /// which is serial; the closure that reads it runs synchronously inside
    /// the same call.
    private final class Busy: @unchecked Sendable { var value = false }

    // MARK: - the arithmetic between the GEMVs

    @Test func rmsNormScalesToUnitRootMeanSquare() {
        var values: [Float] = [3, 4, 0, 0]
        CPUOps.rmsNorm(&values, gamma: [1, 1, 1, 1], epsilon: 0)
        let mean = values.reduce(0) { $0 + $1 * $1 } / 4
        #expect(abs(mean - 1) < 1e-5)
    }

    /// A per-head norm takes each head's own statistics. Sharing one across
    /// heads is a plausible-looking bug that changes every attention output.
    @Test func perSliceNormUsesEachSlicesOwnStatistics() {
        var values: [Float] = [1, 1, 100, 100]
        CPUOps.rmsNormPerSlice(&values, width: 2, gamma: [1, 1], epsilon: 0)
        #expect(abs(values[0] - 1) < 1e-4)
        #expect(abs(values[2] - 1) < 1e-4, "the large slice normalizes to the same place")
    }

    @Test func l2NormalizeMakesUnitSlices() {
        var values: [Float] = [3, 4, 6, 8]
        CPUOps.l2NormalizePerSlice(&values, width: 2, epsilon: 0)
        #expect(abs((values[0] * values[0] + values[1] * values[1]) - 1) < 1e-5)
        #expect(abs((values[2] * values[2] + values[3] * values[3]) - 1) < 1e-5)
    }

    @Test func softmaxSumsToOneAndIsShiftInvariant() {
        var a: [Float] = [1, 2, 3]
        var b: [Float] = [101, 102, 103]
        CPUOps.softmaxInPlace(&a)
        CPUOps.softmaxInPlace(&b)
        #expect(abs(a.reduce(0, +) - 1) < 1e-6)
        for index in a.indices { #expect(abs(a[index] - b[index]) < 1e-6) }
    }

    /// Position zero is the identity, whatever the constants are — which is
    /// exactly why `rope_theta` and the partial fraction cannot be checked
    /// there, and why sequence parity is a separate gate.
    @Test func ropeAtPositionZeroIsTheIdentity() {
        let original = (0..<8).map { Float($0) }
        var values = original
        CPUOps.applyRoPE(&values, headDim: 8, rotaryDim: 4, position: 0, theta: 10_000)
        #expect(values == original)
    }

    /// Partial rotary: only the first `rotaryDim` dimensions move. Rotating
    /// the whole head is the most plausible way to get a model that runs and
    /// is quietly wrong.
    @Test func ropeLeavesTheUnrotatedTailAlone() {
        var values = (0..<8).map { Float($0 + 1) }
        CPUOps.applyRoPE(&values, headDim: 8, rotaryDim: 4, position: 3, theta: 10_000)
        #expect(values[4...] == ArraySlice((4..<8).map { Float($0 + 1) }))
        #expect(values[0] != 1, "the rotated prefix must actually move")
    }

    /// The pairs are `(i, i + rotaryDim/2)` — a half-split inside the
    /// rotated prefix, not adjacent elements. A rotation preserves the
    /// length of each pair, which is what pins the pairing.
    @Test func ropePairsAcrossTheHalfOfTheRotatedPrefix() {
        var values: [Float] = [1, 0, 0, 0, 9, 9, 9, 9]
        CPUOps.applyRoPE(&values, headDim: 8, rotaryDim: 4, position: 5, theta: 10_000)
        let paired = values[0] * values[0] + values[2] * values[2]
        #expect(abs(paired - 1) < 1e-5, "index 0 rotates against index 2, not index 1")
    }

    @Test func activationsMatchTheirDefinitions() {
        #expect(abs(CPUOps.sigmoid(0) - 0.5) < 1e-6)
        #expect(abs(CPUOps.silu(0)) < 1e-6)
        #expect(abs(CPUOps.silu(1) - 1 / (1 + expf(-1))) < 1e-6)
        // softplus must not overflow where a naive log(1 + e^x) would.
        #expect(abs(CPUOps.softplus(100) - 100) < 1e-3)
        #expect(CPUOps.softplus(-100) >= 0)
        #expect(abs(CPUOps.softplus(0) - logf(2)) < 1e-6)
    }
}

/// The CPU serving path: which architectures it will take, how it samples,
/// and the ceilings it enforces rather than promises.
@Suite struct CPUServingTests {

    /// A snapshot outside the family is refused by name. A forward pass on
    /// the wrong layer shape does not crash — it produces fluent nonsense,
    /// which is the same failure as a bad converter with a longer feedback
    /// loop.
    @Test func onlyKnownArchitecturesAreServed() {
        #expect(CPUModelFamily.resolve(modelType: "qwen3_5_dense") == .qwen35Dense)
        #expect(CPUModelFamily.resolve(modelType: "qwen3_5_text") == .qwen35Dense)
        #expect(CPUModelFamily.resolve(modelType: "llama") == nil)
        #expect(CPUModelFamily.resolve(modelType: nil) == nil)

        let refusal = CPUModelFamily.refusal(modelType: "llama")
        #expect(refusal.contains("llama"), "say which one was refused")
        #expect(refusal.contains("qwen3_5_dense"), "and which are served")
    }

    /// Greedy is the default and has to be exactly greedy: the memory work
    /// compares runs against each other, which only means anything if the
    /// same prompt gives the same answer.
    @Test func greedyPicksTheMaximumEveryTime() {
        let sampler = CPUSampler()
        #expect(sampler.isGreedy)
        let logits: [Float] = [0.1, 5.0, -2, 4.9, 0]
        let generator = sampler.makeGenerator()
        for _ in 0..<20 {
            #expect(sampler.pick(logits, using: generator) == 1)
        }
    }

    /// Top-k of one is greedy however hot the temperature, which is the
    /// property that makes the two knobs composable.
    @Test func topKOfOneIsGreedy() {
        let sampler = CPUSampler(temperature: 2, topP: 1, topK: 1, seed: 7)
        let logits: [Float] = [0.1, 5.0, -2, 4.9, 0]
        let generator = sampler.makeGenerator()
        for _ in 0..<20 {
            #expect(sampler.pick(logits, using: generator) == 1)
        }
    }

    /// Sampling stays inside the distribution it was given: nothing outside
    /// the top-k may ever be chosen, however the random draw falls.
    @Test func samplingNeverEscapesTopK() {
        let sampler = CPUSampler(temperature: 1.5, topP: 1, topK: 2, seed: 99)
        let logits: [Float] = [0.1, 5.0, -2, 4.9, 0]
        let generator = sampler.makeGenerator()
        var seen = Set<Int>()
        for _ in 0..<200 { seen.insert(sampler.pick(logits, using: generator)) }
        #expect(seen.isSubset(of: [1, 3]), "chose from outside the top two: \(seen)")
    }

    /// A seeded sampler repeats itself, so a run can be reproduced.
    @Test func aSeededSamplerIsReproducible() {
        let logits: [Float] = [1, 2, 3, 2, 1]
        func draw() -> [Int] {
            let sampler = CPUSampler(temperature: 1, topP: 1, topK: 0, seed: 42)
            let generator = sampler.makeGenerator()
            return (0..<30).map { _ in sampler.pick(logits, using: generator) }
        }
        #expect(draw() == draw())
    }

    /// Top-p trims by mass. With almost all of it on one token, a small p
    /// leaves only that token.
    @Test func topPKeepsOnlyTheMass() {
        let sampler = CPUSampler(temperature: 0.5, topP: 0.5, topK: 0, seed: 3)
        let logits: [Float] = [0, 10, 0, 0]
        let generator = sampler.makeGenerator()
        for _ in 0..<20 { #expect(sampler.pick(logits, using: generator) == 1) }
    }

}
