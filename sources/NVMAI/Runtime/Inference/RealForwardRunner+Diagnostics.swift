import Foundation
import Metal

/// Diagnostics: NVMAI_KERNEL_STATS timing, route/prefetch traces, and the RDAdvise policy. Never on the production hot path unless the corresponding env switch is set.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes.
extension RealForwardRunner {
    func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    public func resetKernelGPUTimings() {
        kernelGPUTimings.removeAll(keepingCapacity: true)
    }

    func recordKernelGPU(role: String, _ cb: MTLCommandBuffer) {
        guard kernelGPUTimingsEnabled, cb.gpuEndTime > 0 else { return }
        kernelGPUTimings.append(
            KernelGPUTiming(role: role, start: cb.gpuStartTime, end: cb.gpuEndTime))
    }

    /// Aggregated per-role GPU milliseconds for the current generation,
    /// largest first.
    public func kernelGPUTimingSummary() -> [(role: String, millis: Double, count: Int)] {
        var acc: [String: (millis: Double, count: Int)] = [:]
        for t in kernelGPUTimings {
            let millis = (t.end - t.start) * 1000
            acc[t.role, default: (0, 0)].millis += millis
            acc[t.role]!.count += 1
        }
        return acc.map { (role: $0.key, millis: $0.value.millis, count: $0.value.count) }
            .sorted { $0.millis > $1.millis }
    }

    /// Wall-clock span in which *any* recorded command buffer was on the GPU,
    /// and the span from the first start to the last end.
    ///
    /// Per-role sums double-count: the decode path deliberately runs the routed
    /// MoE buffer concurrently with the next layer's attention, so adding the
    /// roles together can exceed the time that actually elapsed. Merging the
    /// intervals answers the question the sums cannot -- whether the GPU is
    /// saturated (busy ~= span, so the only gain left is cheaper kernels) or
    /// idle in the gaps (busy << span, so there is overlap still to win).
    public func kernelGPUOccupancy() -> (busyMillis: Double, spanMillis: Double) {
        guard !kernelGPUTimings.isEmpty else { return (0, 0) }
        let sorted = kernelGPUTimings.sorted { $0.start < $1.start }
        var busy: TimeInterval = 0
        var mergedStart = sorted[0].start
        var mergedEnd = sorted[0].end
        for t in sorted.dropFirst() {
            if t.start > mergedEnd {
                busy += mergedEnd - mergedStart
                mergedStart = t.start
                mergedEnd = t.end
            } else if t.end > mergedEnd {
                mergedEnd = t.end
            }
        }
        busy += mergedEnd - mergedStart
        let span = sorted.map(\.end).max()! - sorted[0].start
        return (busy * 1000, span * 1000)
    }

    /// Where the GPU's idle time actually sits, attributed to the transition
    /// it falls in.
    ///
    /// `kernelGPUOccupancy` says how much idle there is; this says where. Each
    /// gap between one buffer finishing and the next starting is charged to the
    /// pair of roles it separates, so "attn_tail_router -> moe_phase1_2_routed"
    /// accumulates the wait for the router readback and expert fetch, while
    /// "moe_phase1_2_routed -> attn_norm_qkv" accumulates the per-layer
    /// turnaround. Without this the only way to pick a target is to divide
    /// total idle by a buffer count and assume the quotient means something,
    /// which is exactly the reasoning that produced a failed optimisation.
    public func kernelGPUGaps() -> [(transition: String, millis: Double, count: Int)] {
        guard kernelGPUTimings.count > 1 else { return [] }
        let sorted = kernelGPUTimings.sorted { $0.start < $1.start }
        var acc: [String: (millis: Double, count: Int)] = [:]
        var previous = sorted[0]
        for current in sorted.dropFirst() {
            // Overlapping buffers contribute no gap; advance the frontier to
            // whichever end is later so a long buffer does not manufacture one.
            let gap = current.start - previous.end
            if gap > 0 {
                let key = "\(previous.role)->\(current.role)"
                acc[key, default: (0, 0)].millis += gap * 1000
                acc[key]!.count += 1
            }
            if current.end > previous.end { previous = current }
        }
        return acc.map { (transition: $0.key, millis: $0.value.millis, count: $0.value.count) }
            .sorted { $0.millis > $1.millis }
    }

    /// Appends `position layer e0 e1 ... e7` for one decode layer.
    ///
    /// Which experts a token actually routes to is the input to every question
    /// about how expert weights should reach the GPU -- how large the working
    /// set really is, how much reuse there is between consecutive tokens, and
    /// therefore whether a residency scheme that is not the slot cache could
    /// hold it. Synthetic access patterns answer none of that: a full sweep of
    /// the expert file measures thrash that decode never causes, and a random
    /// pattern measures the opposite. This dumps the real thing so a replay can
    /// be driven by it.
    ///
    /// Off unless `NVMAI_ROUTE_TRACE` names a file. Diagnostic only.
    func recordRouteTrace(layer: Int, position: Int, experts: [Int]) {
        guard routeTraceFD >= 0 else { return }
        var line = "\(position) \(layer)"
        for expert in experts { line += " \(expert)" }
        line += "\n"
        let bytes = Array(line.utf8)
        var written = 0
        while written < bytes.count {
            let n = bytes.withUnsafeBytes { raw -> Int in
                write(routeTraceFD, raw.baseAddress!.advanced(by: written),
                      bytes.count - written)
            }
            if n <= 0 { break }
            written += n
        }
    }

    /// Appends one exact pre-plan routing/cache observation. `resident` is
    /// captured before cache planning so a later miss reservation cannot make
    /// the trace falsely report an expert as absent.
    func recordPrefetchTrace(layer: Int,
                                     position: Int,
                                     experts: [Int],
                                     misses: [Int],
                                     resident: [Int],
                                     nextLayerPrediction: [Int],
                                     next2LayerPrediction: [Int] = []) {
        guard prefetchTraceFD >= 0 else { return }
        let line = "{\"position\":\(position),\"layer\":\(layer),\"experts\":\(experts),\"misses\":\(misses),\"resident\":\(resident),\"next_layer_prediction\":\(nextLayerPrediction),\"next2_layer_prediction\":\(next2LayerPrediction)}\n"
        let bytes = Array(line.utf8)
        var written = 0
        while written < bytes.count {
            let count = bytes.withUnsafeBytes { raw -> Int in
                write(prefetchTraceFD, raw.baseAddress!.advanced(by: written),
                      bytes.count - written)
            }
            if count <= 0 { break }
            written += count
        }
    }

    func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            // Skip window is inclusive of `position`, matching the adaptive
            // policy (`position <= skipUntilPosition`), so both policies
            // suppress advice for the same token window after a slow call.
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }
}
