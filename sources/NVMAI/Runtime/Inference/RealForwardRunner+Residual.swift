import Foundation
import Metal

/// How a family moves data between the residual stream and a sublayer.
///
/// The Qwen3.5-MoE families use plain pre-norm: normalize the residual into a
/// block's input, add the block's output back. Every layer does this twice,
/// once around attention and once around the MLP.
///
/// Qwen3.8-Flash-Next replaces both halves. Its residual is four parallel
/// 2560-wide streams, read through a learned gate that collapses them to one
/// vector and written through another that injects the block's output into
/// every stream with its own weight (`HyperConnection`). The shape of the
/// layer loop is unchanged — still normalize-in, combine-out — so the
/// difference belongs at this seam rather than branching the loop.
///
/// These four call sites are the entire coupling. Keeping them in one file
/// means adding the new family's behaviour touches this file and not the
/// decode or prefill bodies, matching how `TensorSchema` isolates naming.
extension RealForwardRunner {
    /// Residual streams this model carries. One for pre-norm families; the
    /// hyper-connection families carry `hc_count`.
    var residualStreamCount: Int {
        cfg.hyperConnections.enabled ? cfg.hyperConnections.count : 1
    }

    /// Width of the residual buffer in elements.
    var residualWidth: Int { cfg.hiddenSize * residualStreamCount }

    /// Residual -> block input, for one decode token.
    func encodeResidualEntryDecode(commandBuffer: MTLCommandBuffer,
                                   hidden: MTLBuffer,
                                   norm: TensorView,
                                   out: MTLBuffer,
                                   eps: Float) throws {
        // Pre-norm. The hyper-connection read gate lands here, keyed off
        // `cfg.hyperConnections.enabled`, once its layer weights are wired.
        try rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            weight: norm.buffer, weightOffset: Int(norm.offset),
                            out: out,
                            d: UInt32(cfg.hiddenSize), eps: eps)
    }

    /// Block output -> residual, for one decode token.
    func encodeResidualExitDecode(commandBuffer: MTLCommandBuffer,
                                  hidden: MTLBuffer,
                                  delta: MTLBuffer) throws {
        // Plain add. The hyper-connection write gate lands here.
        try elementwise!.encodeResidualAdd(commandBuffer: commandBuffer,
                                           hidden: hidden,
                                           delta: delta,
                                           count: cfg.hiddenSize)
    }

    /// Residual -> block input, for a prefill chunk of `tokens` rows.
    func encodeResidualEntryPrefill(commandBuffer: MTLCommandBuffer,
                                    hidden: MTLBuffer,
                                    norm: TensorView,
                                    out: MTLBuffer,
                                    tokens: Int,
                                    eps: Float) throws {
        try prefillRMS.encodeBF16W(commandBuffer: commandBuffer,
                                   x: hidden,
                                   weight: norm.buffer,
                                   weightOffset: Int(norm.offset),
                                   out: out,
                                   t: UInt32(tokens),
                                   d: UInt32(cfg.hiddenSize),
                                   eps: eps)
    }

    /// Block output -> residual, for a prefill chunk of `tokens` rows.
    func encodeResidualExitPrefill(commandBuffer: MTLCommandBuffer,
                                   hidden: MTLBuffer,
                                   delta: MTLBuffer,
                                   tokens: Int) throws {
        try elementwise!.encodeResidualAdd(commandBuffer: commandBuffer,
                                           hidden: hidden,
                                           delta: delta,
                                           count: tokens * cfg.hiddenSize)
    }
}
