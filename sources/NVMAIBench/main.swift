import Foundation
import Metal
import NVMAI

/// Kernel micro-benchmarks. Two families:
///
/// 1. The fused QKV GEMV (baseline/bandwidth/unroll2/ulong2): the decode
///    attention block's dominant kernel in isolation (synthetic buffers, no
///    model pipeline). Reports GB/s against the M3's theoretical peak
///    (~100 GB/s).
///
/// 2. The routed-MoE decode kernels (moe_phase1 / moe_phase2 / moe): the
///    phase-1 gate/up GEMV and the phase-2 down+reduce GEMV with synthetic
///    8-expert blobs at the real 4-bit shapes (F=512, D=2816, topK=8),
///    matching the decode routedCB dispatch. Reports GB/s of weight reads.
///    The decode's routedCB measured 14.4 ms/token (~48 GB/s at the real
///    shapes); this bench isolates which phase and access pattern is slow.
///
/// Usage: NVMAIBench [kernelName] [iterations]
///   qkv family: baseline (default), bandwidth, unroll2, ulong2
///   moe family: moe_phase1, moe_phase2, moe
///   gdn family: gdn_inproj (baseline), gdn_inproj_xsh, gdn_inproj_r16
@main
struct NVMAIBench {
    /// Shader-side `ExpertOffsets` mirror: 9 packed UInt32 in the same order.
    private struct MoEBenchOffsets {
        var gateW: UInt32
        var gateS: UInt32
        var gateB: UInt32
        var upW: UInt32
        var upS: UInt32
        var upB: UInt32
        var downW: UInt32
        var downS: UInt32
        var downB: UInt32
    }

    static func main() throws {
        let kernelName = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1] : "baseline"
        let iterations = CommandLine.arguments.count > 2
            ? Int(CommandLine.arguments[2]) ?? 300 : 300

        let context = try MetalContext()
        let device = context.device
        print("device: \(device.name)")

        if kernelName.hasPrefix("gdn") {
            try runGDN(kernelName: kernelName, iterations: iterations, context: context)
            return
        }

        if kernelName.hasPrefix("moe") {
            try runMoE(kernelName: kernelName, iterations: iterations, context: context)
            return
        }

        // Gated QKV shape (the dominant decode GEMV): qRows = 2*qDim = 8192,
        // kvRows = 2048, n = 2816.
        let qRows: UInt32 = 8192
        let kvRows: UInt32 = 2048
        let n: UInt32 = 2816
        let rowBytes = Int(n) / 2
        let groupCount = Int(n) / 64

        let pso = try context.pipeline(
            kernelName == "baseline" ? "dequant_int4_qkv_gemv_simd"
                : "dequant_int4_qkv_gemv_simd_\(kernelName)",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        func makeBuffer(_ bytes: Int, _ value: UInt8) -> MTLBuffer {
            let buf = device.makeBuffer(length: bytes,
                                        options: .storageModeShared)!
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let qW = makeBuffer(Int(qRows) * rowBytes, 0x12)
        let qS = makeBuffer(Int(qRows) * groupCount * 2, 0x01)
        let qB = makeBuffer(Int(qRows) * groupCount * 2, 0x00)
        let kW = makeBuffer(Int(kvRows) * rowBytes, 0x34)
        let kS = makeBuffer(Int(kvRows) * groupCount * 2, 0x01)
        let kB = makeBuffer(Int(kvRows) * groupCount * 2, 0x00)
        let vW = makeBuffer(Int(kvRows) * rowBytes, 0x56)
        let vS = makeBuffer(Int(kvRows) * groupCount * 2, 0x01)
        let vB = makeBuffer(Int(kvRows) * groupCount * 2, 0x00)
        let x = makeBuffer(Int(n) * 2, 0x77)
        let qOut = makeBuffer(Int(qRows) * 2, 0)
        let kOut = makeBuffer(Int(kvRows) * 2, 0)
        let vOut = makeBuffer(Int(kvRows) * 2, 0)

        let totalRows = Int(qRows + 2 * kvRows)
        let rowsPerThreadgroup = 8
        let threadgroups = (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        let bytesPerLaunch = UInt64(totalRows) * UInt64(rowBytes)

        var qVar = qRows
        var kvVar = kvRows
        var nVar = n

        let cb = context.queue.makeCommandBuffer()!
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qW, offset: 0, index: 0)
        enc.setBuffer(qS, offset: 0, index: 1)
        enc.setBuffer(qB, offset: 0, index: 2)
        enc.setBuffer(kW, offset: 0, index: 3)
        enc.setBuffer(kS, offset: 0, index: 4)
        enc.setBuffer(kB, offset: 0, index: 5)
        enc.setBuffer(vW, offset: 0, index: 6)
        enc.setBuffer(vS, offset: 0, index: 7)
        enc.setBuffer(vB, offset: 0, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        enc.setBuffer(qOut, offset: 0, index: 10)
        enc.setBuffer(kOut, offset: 0, index: 11)
        enc.setBuffer(vOut, offset: 0, index: 12)
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        for _ in 0..<iterations {
            enc.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: rowsPerThreadgroup * 32,
                                               height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let gbPerSec = Double(bytesPerLaunch) / perIteration / 1_000_000_000
        let theoretical = 100.0
        print("kernel=\(kernelName) iterations=\(iterations) "
            + "total=\(String(format: "%.4f", totalSeconds))s "
            + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print("bytes/launch=\(bytesPerLaunch) "
            + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
            + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak")
    }

    /// Routed-MoE decode kernels at the real 4-bit shapes. Weights are the
    /// metric: phase-1 reads gate+up (8 x 2 x F*D/2 bytes), phase-2 reads
    /// down (8 x D*F/2 bytes). The combined "moe" mode dispatches both in
    /// one command buffer, mirroring the decode routedCB.
    private static func runMoE(kernelName: String,
                               iterations: Int,
                               context: MetalContext) throws {
        let device = context.device
        let D: UInt32 = 2048   // qwen36_35B_A3B hiddenSize
        let F: UInt32 = 512    // moeIntermediateSize
        let topK: UInt32 = 8
        let groupCount = Int(D) / 64   // kMoEGroupSize = 64 elements

        // Per-blob 4-bit layout: gate_W, gate_s, gate_b, up_W, up_s, up_b,
        // down_W, down_s, down_b.
        let fD2 = Int(F) * Int(D) / 2
        let groups = Int(F) * groupCount * 2
        let gateWOff = 0
        let gateSOff = fD2
        let gateBOff = gateSOff + groups
        let upWOff = gateBOff + groups
        let upSOff = upWOff + fD2
        let upBOff = upSOff + groups
        let downWOff = upBOff + groups
        let downSOff = downWOff + Int(D) * Int(F) / 2
        let downBOff = downSOff + groups
        let blobBytes = downBOff + groups

        func makeBuffer(_ bytes: Int, _ value: UInt8) -> MTLBuffer {
            let buf = device.makeBuffer(length: bytes,
                                        options: .storageModeShared)!
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        var blobs: [MTLBuffer] = []
        for i in 0..<Int(topK) {
            blobs.append(makeBuffer(blobBytes, UInt8(0x11 + i)))
        }
        // Non-uniform activation pattern so staging/indexing bugs surface:
        // the uniform fill used before masked wrong-element reads.
        let x = device.makeBuffer(length: Int(D) * 2, options: .storageModeShared)!
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(D) {
            xPtr[i] = Float16(Float(i + 1) * 0.001).bitPattern
        }
        let acts = makeBuffer(Int(topK) * Int(F) * 2, 0)
        let routingW = makeBuffer(Int(topK) * 2, 0x3F)
        let residual = makeBuffer(Int(D) * 2, 0x33)
        let y = makeBuffer(Int(D) * 2, 0)

        // RoutedBlobs arg buffer: 8 device pointers (shared memory, so the
        // host address is the GPU address).
        guard let argBuf = device.makeBuffer(length: Int(topK) * 8,
                                             options: .storageModeShared) else {
            fatalError("arg buffer alloc failed")
        }
        let argPtr = argBuf.contents().assumingMemoryBound(to: UnsafeMutableRawPointer?.self)
        for i in 0..<Int(topK) {
            argPtr[i] = blobs[i].contents()
        }

        var offsets = MoEBenchOffsets(
            gateW: UInt32(gateWOff), gateS: UInt32(gateSOff), gateB: UInt32(gateBOff),
            upW: UInt32(upWOff), upS: UInt32(upSOff), upB: UInt32(upBOff),
            downW: UInt32(downWOff), downS: UInt32(downSOff), downB: UInt32(downBOff))

        // Variant dispatch: r4/r16 change rows-per-threadgroup, xsh8/16 stage
        // the activation in threadgroup memory. The production kernel is the
        // 16-row threadgroup-staged layout, so all modes dispatch 16 rows.
        let phase1Variant: String?
        let phase1RowsPerTG: Int
        switch kernelName {
        case "moe_phase1_r8":
            phase1Variant = "moe_phase1_gate_up_act_u16load_r8"
            phase1RowsPerTG = 8
        case "moe_phase1_r16":
            phase1Variant = "moe_phase1_gate_up_act_u16load_r16"
            phase1RowsPerTG = 16
        case "moe_phase1_xsh8":
            phase1Variant = "moe_phase1_gate_up_act_u16load"
            phase1RowsPerTG = 16
        case "moe_phase1_xsh16":
            phase1Variant = "moe_phase1_gate_up_act_u16load"
            phase1RowsPerTG = 16
        default:
            // The production kernel is the 16-row threadgroup-staged layout.
            phase1Variant = nil
            phase1RowsPerTG = 16
        }
        let phase1Kernel = phase1Variant ?? "moe_phase1_gate_up_act_u16load"

        let phase1PSO = try context.pipeline(
            phase1Kernel,
            constants: [],
            maxTotalThreadsPerThreadgroup: phase1RowsPerTG * 32)
        let phase2PSO = try context.pipeline(
            "moe_phase2_down_reduce_k8",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        let subsetPSO = try context.pipeline(
            kernelName == "moe_phase1_subset"
                ? "moe_phase1_gate_up_act_subset_u16load"
                : "moe_phase1_gate_up_act_u16load",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        let phase1Groups = (Int(topK) * Int(F) + phase1RowsPerTG - 1) / phase1RowsPerTG
        let phase1Bytes = UInt64(Int(topK) * 2 * Int(F) * Int(D) / 2)
        let phase2Bytes = UInt64(Int(topK) * Int(D) * Int(F) / 2)

        var Dv = D
        var Fv = F
        var TK = topK

        let cb = context.queue.makeCommandBuffer()!
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        let runPhase1 = kernelName == "moe" || kernelName.hasPrefix("moe_phase1")
        let runPhase2 = kernelName == "moe_phase2" || kernelName == "moe"
        let runSubset = kernelName == "moe_phase1_subset"
        // active-slot buffer for the subset mode (all 8 experts active).
        var activeSlots = [UInt32](0..<topK)
        let activeSlotsBuf = device.makeBuffer(length: Int(topK) * MemoryLayout<UInt32>.size,
                                               options: .storageModeShared)!
        activeSlotsBuf.contents().copyMemory(from: &activeSlots,
                                             byteCount: Int(topK) * MemoryLayout<UInt32>.size)
        var activeCount = topK
        for _ in 0..<iterations {
            if runSubset {
                enc.setComputePipelineState(subsetPSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 5)
                enc.setBytes(&TK, length: MemoryLayout<UInt32>.size, index: 6)
                enc.setBuffer(activeSlotsBuf, offset: 0, index: 7)
                enc.setBytes(&activeCount, length: MemoryLayout<UInt32>.size, index: 8)
                enc.dispatchThreadgroups(
                    MTLSize(width: (Int(activeCount * F) + 15) / 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            }
            if runPhase1 {
                enc.setComputePipelineState(phase1PSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 5)
                enc.setBytes(&TK, length: MemoryLayout<UInt32>.size, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: phase1Groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: phase1RowsPerTG * 32,
                                                   height: 1, depth: 1))
            }
            if runPhase2 {
                enc.setComputePipelineState(phase2PSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(acts, offset: 0, index: 2)
                enc.setBuffer(routingW, offset: 0, index: 3)
                enc.setBuffer(residual, offset: 0, index: 4)
                enc.setBuffer(y, offset: 0, index: 5)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 6)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(D), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("COMMAND BUFFER ERROR: \(err)")
        }

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let bytes: UInt64 = (runPhase1 ? phase1Bytes : 0) + (runPhase2 ? phase2Bytes : 0)
        let gbPerSec = Double(bytes) / perIteration / 1_000_000_000
        let theoretical = 100.0
        // Correctness probe: FNV-1a over the acts buffer (the phase-1 output)
        // and the y buffer (the phase-2 output).
        var hash: UInt32 = 0x811c9dc5
        let actsPtr = acts.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<min(acts.length, 8192) {
            hash ^= UInt32(actsPtr[i])
            hash &*= 0x01000193
        }
        var yHash: UInt32 = 0x811c9dc5
        let yPtr = y.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<min(y.length, 8192) {
            yHash ^= UInt32(yPtr[i])
            yHash &*= 0x01000193
        }
        let actsHalf = acts.contents().assumingMemoryBound(to: UInt16.self)
        let sample = (0..<min(16, acts.length / 2)).map { String(format: "%04x", actsHalf[$0]) }.joined(separator: " ")
        print("kernel=\(kernelName) iterations=\(iterations) "
            + "total=\(String(format: "%.4f", totalSeconds))s "
            + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print("bytes/launch=\(bytes) (phase1=\(phase1Bytes) phase2=\(phase2Bytes)) "
            + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
            + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak "
            + "acts_fnv=\(String(format: "%08x", hash)) y_fnv=\(String(format: "%08x", yHash)) "
            + "acts_sample=\(sample)")
    }

    /// GDN fused input-projection GEMV at the real qwen36 shapes
    /// (qkvDim 8192, valueDim 4096, ab 32 each, N=2048). The weight read is
    /// the metric: qkv + z + a + b ~= 12.7 MB/layer. Variants:
    /// gdn_inproj (production 8-row), gdn_inproj_xsh8 (8-row + tgmem x),
    /// gdn_inproj_r16 (16-row device x), gdn_inproj_xsh16 (16-row + tgmem).
    private static func runGDN(kernelName: String,
                               iterations: Int,
                               context: MetalContext) throws {
        let device = context.device
        let qkvRows: UInt32 = 8192
        let zRows: UInt32 = 4096
        let abRows: UInt32 = 32
        let N: UInt32 = 2048
        let groupCount = Int(N) / 64

        func makeBuffer(_ bytes: Int, _ value: UInt8) -> MTLBuffer {
            let buf = device.makeBuffer(length: bytes, options: .storageModeShared)!
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let qkvW = makeBuffer(Int(qkvRows) * Int(N) / 2, 0x12)
        let qkvS = makeBuffer(Int(qkvRows) * groupCount * 2, 0x01)
        let qkvB = makeBuffer(Int(qkvRows) * groupCount * 2, 0x00)
        let zW = makeBuffer(Int(zRows) * Int(N) / 2, 0x34)
        let zS = makeBuffer(Int(zRows) * groupCount * 2, 0x01)
        let zB = makeBuffer(Int(zRows) * groupCount * 2, 0x00)
        let aW = makeBuffer(Int(abRows) * Int(N) / 2, 0x56)
        let aS = makeBuffer(Int(abRows) * groupCount * 2, 0x01)
        let aB = makeBuffer(Int(abRows) * groupCount * 2, 0x00)
        let bW = makeBuffer(Int(abRows) * Int(N) / 2, 0x78)
        let bS = makeBuffer(Int(abRows) * groupCount * 2, 0x01)
        let bB = makeBuffer(Int(abRows) * groupCount * 2, 0x00)
        let x = device.makeBuffer(length: Int(N) * 2, options: .storageModeShared)!
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(N) {
            xPtr[i] = Float16(Float(i + 1) * 0.001).bitPattern
        }
        let qkvY = makeBuffer(Int(qkvRows) * 2, 0)
        let zY = makeBuffer(Int(zRows) * 2, 0)
        let aY = makeBuffer(Int(abRows) * 2, 0)
        let bY = makeBuffer(Int(abRows) * 2, 0)

        let kernelName2: String
        let rowsPerTG: Int
        switch kernelName {
        case "gdn_inproj_xsh8":
            kernelName2 = "gdn_in_proj_gemv_simd_xsh8"
            rowsPerTG = 8
        case "gdn_inproj_r16":
            kernelName2 = "gdn_in_proj_gemv_simd_r16"
            rowsPerTG = 16
        case "gdn_inproj_xsh16":
            kernelName2 = "gdn_in_proj_gemv_simd_xsh16"
            rowsPerTG = 16
        default:
            kernelName2 = "gdn_in_proj_gemv_simd"
            rowsPerTG = 8
        }
        let pso = try context.pipeline(
            kernelName2, constants: [],
            maxTotalThreadsPerThreadgroup: rowsPerTG * 32)

        let totalRows = Int(qkvRows + zRows + 2 * abRows)
        let bytes = UInt64(Int(qkvRows) * Int(N) / 2 + Int(zRows) * Int(N) / 2
                           + 2 * Int(abRows) * Int(N) / 2)
        var qkvVar = qkvRows
        var zVar = zRows
        var abVar = abRows
        var nVar = N

        let cb = context.queue.makeCommandBuffer()!
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qkvW, offset: 0, index: 0)
        enc.setBuffer(qkvS, offset: 0, index: 1)
        enc.setBuffer(qkvB, offset: 0, index: 2)
        enc.setBuffer(zW, offset: 0, index: 3)
        enc.setBuffer(zS, offset: 0, index: 4)
        enc.setBuffer(zB, offset: 0, index: 5)
        enc.setBuffer(aW, offset: 0, index: 6)
        enc.setBuffer(aS, offset: 0, index: 7)
        enc.setBuffer(aB, offset: 0, index: 8)
        enc.setBuffer(bW, offset: 0, index: 9)
        enc.setBuffer(bS, offset: 0, index: 10)
        enc.setBuffer(bB, offset: 0, index: 11)
        enc.setBuffer(x, offset: 0, index: 12)
        enc.setBuffer(qkvY, offset: 0, index: 13)
        enc.setBuffer(zY, offset: 0, index: 14)
        enc.setBuffer(aY, offset: 0, index: 15)
        enc.setBuffer(bY, offset: 0, index: 16)
        enc.setBytes(&qkvVar, length: MemoryLayout<UInt32>.size, index: 17)
        enc.setBytes(&zVar, length: MemoryLayout<UInt32>.size, index: 18)
        enc.setBytes(&abVar, length: MemoryLayout<UInt32>.size, index: 19)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 20)
        for _ in 0..<iterations {
            enc.dispatchThreadgroups(
                MTLSize(width: (totalRows + rowsPerTG - 1) / rowsPerTG, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: rowsPerTG * 32, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("COMMAND BUFFER ERROR: \(err)")
        }

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let gbPerSec = Double(bytes) / perIteration / 1_000_000_000
        let theoretical = 100.0
        print("kernel=\(kernelName) iterations=\(iterations) "
            + "total=\(String(format: "%.4f", totalSeconds))s "
            + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print("bytes/launch=\(bytes) "
            + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
            + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak")
    }
}
