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
