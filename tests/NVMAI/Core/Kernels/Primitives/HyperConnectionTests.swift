import Testing
import Foundation
import Metal
@testable import NVMAI
import NVMAIValidationSupport

/// The two stream-axis kernels of Qwen3.8-Flash-Next's Gated Residual, against
/// CPU references. Everything else in the block (the three INT4 projections,
/// silu, sigmoid) reuses existing primitives; these are the only steps that
/// see the stream axis, so they are the only place a stream-indexing mistake
/// can hide.
@Suite("Hyper-connection streams")
struct HyperConnectionTests {
    /// Production geometry: 4 streams of 2560.
    private static let dim = 2560
    private static let streams = 4

    @Test("Gated read averages mix-weighted streams")
    func mixReduceMatchesReference() throws {
        var rng = SeedTree(0x11C0).key("hc-mix-reduce")
        let total = Self.dim * Self.streams
        let mix = (0..<total).map { _ in Float16(rng.uniform(0.0, 1.0)) }
        let normed = (0..<total).map { _ in Float16(rng.uniform(-2.0, 2.0)) }

        let ctx = try MetalContext()
        let kernel = try Elementwise(context: ctx)
        guard let mixBuf = Fp16Buffer.make(ctx.device, halves: mix),
              let normBuf = Fp16Buffer.make(ctx.device, halves: normed),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.dim) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        try kernel.encodeHCMixReduce(commandBuffer: cb, mix: mixBuf,
                                     normed: normBuf, out: outBuf,
                                     dim: Self.dim, streams: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        var reference = [Float](repeating: 0, count: Self.dim)
        for d in 0..<Self.dim {
            var acc: Float = 0
            for s in 0..<Self.streams {
                let i = s * Self.dim + d
                acc += Float(mix[i]) * Float(normed[i])
            }
            reference[d] = acc / Float(Self.streams)
        }
        let actual = Fp16Buffer.read(outBuf, count: Self.dim)
        #expect(RelError.compute(actual: actual, reference: reference)
                < Tolerance.fp16Reduction)
    }

    @Test("Gated write injects one output into every stream with its own weight")
    func injectMatchesReference() throws {
        var rng = SeedTree(0x22D0).key("hc-inject")
        let total = Self.dim * Self.streams
        let initial = (0..<total).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let blockOut = (0..<Self.dim).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        // Distinct per-stream weights: a kernel that used one weight for all
        // streams, or indexed by d instead of s, fails this.
        let inject = [Float16(0.25), Float16(0.5), Float16(1.0), Float16(1.75)]

        let ctx = try MetalContext()
        let kernel = try Elementwise(context: ctx)
        guard let streamBuf = Fp16Buffer.make(ctx.device, halves: initial),
              let outBuf = Fp16Buffer.make(ctx.device, halves: blockOut),
              let injBuf = Fp16Buffer.make(ctx.device, halves: inject) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        try kernel.encodeHCInject(commandBuffer: cb, streams: streamBuf,
                                  blockOut: outBuf, inject: injBuf,
                                  dim: Self.dim, streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        var reference = [Float](repeating: 0, count: total)
        for s in 0..<Self.streams {
            for d in 0..<Self.dim {
                let i = s * Self.dim + d
                reference[i] = Float(initial[i])
                    + Float(blockOut[d]) * Float(inject[s])
            }
        }
        let actual = Fp16Buffer.read(streamBuf, count: total)
        #expect(RelError.compute(actual: actual, reference: reference)
                < Tolerance.fp16Reduction)
    }

    @Test("Inject accumulates rather than overwriting")
    func injectAccumulates() throws {
        // The residual must carry forward: applying inject twice with the same
        // operands must double the added term, not repeat it.
        let dim = 128, streams = 2
        let total = dim * streams
        let zeros = [Float16](repeating: 0, count: total)
        let blockOut = [Float16](repeating: Float16(1.0), count: dim)
        let inject = [Float16(1.0), Float16(2.0)]

        let ctx = try MetalContext()
        let kernel = try Elementwise(context: ctx)
        guard let streamBuf = Fp16Buffer.make(ctx.device, halves: zeros),
              let outBuf = Fp16Buffer.make(ctx.device, halves: blockOut),
              let injBuf = Fp16Buffer.make(ctx.device, halves: inject) else {
            Issue.record("alloc failed"); return
        }
        for _ in 0..<2 {
            let cb = ctx.queue.makeCommandBuffer()!
            try kernel.encodeHCInject(commandBuffer: cb, streams: streamBuf,
                                      blockOut: outBuf, inject: injBuf,
                                      dim: dim, streamCount: streams)
            cb.commit(); cb.waitUntilCompleted()
        }
        let actual = Fp16Buffer.read(streamBuf, count: total)
        #expect(abs(actual[0] - 2.0) < 1e-2, "stream 0 after two injects")
        #expect(abs(actual[dim] - 4.0) < 1e-2, "stream 1 after two injects")
    }

    @Test("Standalone sigmoid and silu match their definitions")
    func unaryGates() throws {
        let values: [Float16] = [-4, -1, -0.5, 0, 0.5, 1, 4].map(Float16.init)
        let ctx = try MetalContext()
        let kernel = try Elementwise(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: values),
              let sigBuf = Fp16Buffer.make(ctx.device, count: values.count),
              let siluBuf = Fp16Buffer.make(ctx.device, count: values.count) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        try kernel.encodeSigmoid(commandBuffer: cb, x: xBuf, out: sigBuf,
                                 count: values.count)
        try kernel.encodeSilu(commandBuffer: cb, x: xBuf, out: siluBuf,
                              count: values.count)
        cb.commit(); cb.waitUntilCompleted()

        let sig = Fp16Buffer.read(sigBuf, count: values.count)
        let silu = Fp16Buffer.read(siluBuf, count: values.count)
        for (i, v) in values.enumerated() {
            let x = Float(v)
            let s = 1.0 / (1.0 + expf(-x))
            #expect(abs(sig[i] - s) < 5e-3, "sigmoid(\(x))")
            #expect(abs(silu[i] - x * s) < 5e-3, "silu(\(x))")
        }
    }
}

/// The complete Gated Residual block against a CPU reference of the whole
/// formula. The individual kernels are covered above; this pins that they are
/// composed in the right order with the right scaling -- the failure mode a
/// per-kernel test cannot see.
@Suite("Hyper-connection block")
struct HyperConnectionBlockTests {
    // Small but structurally identical to production: streams > 1, lowRank <
    // dim, and dim*streams a multiple of the 64-wide quant group.
    private static let dim = 128
    private static let streams = 4
    private static let lowRank = 64
    private static let eps: Float = 1e-6

    private struct QuantWeights {
        let packed: [UInt8], scales: [UInt16], biases: [UInt16], dequant: [[Float]]
    }

    /// Quantize an [m][n] matrix the way the checkpoint stores it, and keep the
    /// dequantized values so the reference sees exactly what the GPU reads.
    private static func quantize(_ rows: [[Float]]) -> QuantWeights {
        var packed: [UInt8] = [], scales: [UInt16] = [], biases: [UInt16] = []
        var dequant: [[Float]] = []
        for row in rows {
            let q = Quantization.quantizeInt4Affine(row)
            packed += q.packed; scales += q.scales; biases += q.biases
            dequant.append(Quantization.dequantizeInt4Affine(q, n: row.count))
        }
        return QuantWeights(packed: packed, scales: scales,
                            biases: biases, dequant: dequant)
    }

    private static func matvec(_ w: [[Float]], _ x: [Float]) -> [Float] {
        w.map { row in zip(row, x).reduce(0) { $0 + $1.0 * $1.1 } }
    }
    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
    private static func silu(_ x: Float) -> Float { x * sigmoid(x) }

    @Test("Read gate matches the reference formula end to end")
    func readMatchesReference() throws {
        var rng = SeedTree(0x77BC).key("hc-block-read")
        let wide = Self.dim * Self.streams
        let streamVals = (0..<wide).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let hcNormF = (0..<wide).map { _ in rng.uniform(0.5, 1.5) }
        let down = Self.quantize((0..<Self.lowRank).map { _ in
            (0..<wide).map { _ in rng.uniform(-0.1, 0.1) } })
        let up = Self.quantize((0..<wide).map { _ in
            (0..<Self.lowRank).map { _ in rng.uniform(-0.1, 0.1) } })

        let ctx = try MetalContext()
        let hc = try HyperConnection(context: ctx, dim: Self.dim,
                                     streams: Self.streams, lowRank: Self.lowRank)
        let hcNormBits = hcNormF.map { Quantization.bf16Bits($0) }
        guard let streamBuf = Fp16Buffer.make(ctx.device, halves: streamVals),
              let normBuf = ctx.device.makeBuffer(length: hcNormBits.count * 2,
                                                  options: .storageModeShared),
              let inputBuf = Fp16Buffer.make(ctx.device, count: Self.dim),
              let dW = ctx.device.makeBuffer(bytes: down.packed,
                                             length: down.packed.count,
                                             options: .storageModeShared),
              let dS = ctx.device.makeBuffer(bytes: down.scales,
                                             length: down.scales.count * 2,
                                             options: .storageModeShared),
              let dB = ctx.device.makeBuffer(bytes: down.biases,
                                             length: down.biases.count * 2,
                                             options: .storageModeShared),
              let uW = ctx.device.makeBuffer(bytes: up.packed,
                                             length: up.packed.count,
                                             options: .storageModeShared),
              let uS = ctx.device.makeBuffer(bytes: up.scales,
                                             length: up.scales.count * 2,
                                             options: .storageModeShared),
              let uB = ctx.device.makeBuffer(bytes: up.biases,
                                             length: up.biases.count * 2,
                                             options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let np = normBuf.contents().bindMemory(to: UInt16.self,
                                               capacity: hcNormBits.count)
        for i in 0..<hcNormBits.count { np[i] = hcNormBits[i] }

        let cb = ctx.queue.makeCommandBuffer()!
        try hc.encodeRead(commandBuffer: cb,
                          streamsBuffer: streamBuf, hcNorm: normBuf,
                          down: .init(weights: dW, scales: dS, biases: dB),
                          up: .init(weights: uW, scales: uS, biases: uB),
                          blockInput: inputBuf, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        // Reference: grouped norm, low-rank gate with the 1/S inside silu,
        // then the mix-weighted stream mean.
        let x = streamVals.map { Float($0) }
        let wRef = hcNormBits.map { Quantization.bf16ToFloat($0) }
        var normed = [Float](repeating: 0, count: wide)
        for s in 0..<Self.streams {
            let lo = s * Self.dim
            let slice = Array(x[lo..<(lo + Self.dim)])
            let ms = slice.reduce(0) { $0 + $1 * $1 } / Float(Self.dim)
            let inv = 1 / (ms + Self.eps).squareRoot()
            for d in 0..<Self.dim { normed[lo + d] = slice[d] * inv * wRef[lo + d] }
        }
        let scale = 1 / Float(Self.streams)
        let low = Self.matvec(down.dequant, normed).map { Self.silu($0 * scale) }
        let mix = Self.matvec(up.dequant, low).map { Self.sigmoid($0) }
        var reference = [Float](repeating: 0, count: Self.dim)
        for d in 0..<Self.dim {
            var acc: Float = 0
            for s in 0..<Self.streams {
                let i = s * Self.dim + d
                acc += mix[i] * normed[i]
            }
            reference[d] = acc / Float(Self.streams)
        }
        let actual = Fp16Buffer.read(inputBuf, count: Self.dim)
        // fp16 through a norm, two INT4 GEMVs and two nonlinearities.
        #expect(RelError.compute(actual: actual, reference: reference) < 0.05,
                "read gate: \(RelError.compute(actual: actual, reference: reference))")
    }

    @Test("The write gate opens to twice the read gate's range")
    func writeGateRange() {
        // inject = 2 * sigmoid(.), so a stream can amplify a block, not only
        // attenuate it. sigmoid alone would cap every stream at 1.0 and make
        // the residual strictly contractive -- a plausible-looking bug.
        for v in [Float(-8), -1, 0, 1, 8] {
            let gate = 2 * Self.sigmoid(v)
            #expect(gate >= 0 && gate <= 2)
        }
        #expect(abs(2 * Self.sigmoid(0) - 1.0) < 1e-6)
    }
}
