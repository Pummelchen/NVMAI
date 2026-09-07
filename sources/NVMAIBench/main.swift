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

        // Before the Metal context: these measure the CPU and must not
        // touch the GPU, whose bandwidth is the thing being competed for.
        if try runCPUCommand(kernelName, iterations: iterations) { return }

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

        if kernelName.hasPrefix("head") {
            try runHead(kernelName: kernelName, iterations: iterations, context: context)
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
    /// lint:allow-long NVMAIBench is a development harness, not a shipped
    /// product: each run* is one linear measurement script whose setup,
    /// dispatch and reporting only make sense read top to bottom.
    private static func runMoE(kernelName: String,
                               iterations: Int,
                               context: MetalContext) throws {
        let device = context.device
        // Default shape is Qwen 3.6 35B-A3B. NVMAI_BENCH_MOE_SHAPE=qwen38
        // selects Qwen3.8-Flash-Next's routed expert (D 2560, F 640, top-10),
        // the shape the decode profile's moe_* numbers come from.
        let qwen38 = ProcessInfo.processInfo.environment["NVMAI_BENCH_MOE_SHAPE"] == "qwen38"
        let D: UInt32 = qwen38 ? 2560 : 2048   // hiddenSize
        let F: UInt32 = qwen38 ? 640 : 512     // moeIntermediateSize
        let topK: UInt32 = qwen38 ? 10 : 8
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
            let blob = makeBuffer(blobBytes, UInt8(0x11 + i))
            // Scale/bias regions as bf16 0x3C3C (~0.011) rather than the
            // byte pattern (~1e-38), so the activations are non-zero and
            // acts_fnv actually compares kernel variants.
            for off in [gateSOff, gateBOff, upSOff, upBOff, downSOff, downBOff] {
                memset(blob.contents().advanced(by: off), 0x3C, groups)
            }
            blobs.append(blob)
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
        case "moe_phase1_v2":
            // Two rows per simdgroup, 16 simdgroups: 32 rows per 512 threads.
            phase1Variant = "moe_phase1_gate_up_act_u16load_v2"
            phase1RowsPerTG = 32
        default:
            // The production kernel is the 16-row threadgroup-staged layout.
            phase1Variant = nil
            phase1RowsPerTG = 16
        }
        let phase1Threads = kernelName == "moe_phase1_v2" ? 512 : phase1RowsPerTG * 32
        let phase1Kernel = phase1Variant ?? "moe_phase1_gate_up_act_u16load"

        // NVMAI_BENCH_MOE_SPECIALIZE=1 builds the phase-1 pipeline with the
        // runtime's function constants (D, F, top-k, silu, host-gated I/O);
        // the unspecialized kernel reads its shape from buffers and measures
        // ~35 GB/s where the specialized one moves the same blobs at ~60.
        let specialize = ProcessInfo.processInfo.environment["NVMAI_BENCH_MOE_SPECIALIZE"] == "1"
        let phase1Constants: [MetalFunctionConstant] = specialize ? [
            MetalFunctionConstant(index: 0, value: .uint32(D)),
            MetalFunctionConstant(index: 1, value: .uint32(F)),
            MetalFunctionConstant(index: 2, value: .uint32(topK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(true)),
            MetalFunctionConstant(index: 6, value: .bool(false)),
        ] : []
        let phase1PSO = try context.pipeline(
            phase1Kernel,
            constants: phase1Constants,
            maxTotalThreadsPerThreadgroup: phase1Threads)
        let phase2PSO = try context.pipeline(
            topK == 8 ? "moe_phase2_down_reduce_k8" : "moe_phase2_down_reduce_kn",
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
        let phase1TG = MTLSize(width: phase1Threads, height: 1, depth: 1)
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
                    threadsPerThreadgroup: phase1TG)
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
    /// lint:allow-long NVMAIBench is a development harness, not a shipped
    /// product: each run* is one linear measurement script whose setup,
    /// dispatch and reporting only make sense read top to bottom.
    private static func runGDN(kernelName: String,
                               iterations: Int,
                               context: MetalContext) throws {
        let device = context.device
        // Default shape is the Qwen 3.6 GDN layer. NVMAI_BENCH_GDN_SHAPE=qwen38
        // selects Qwen3.8-Flash-Next's (16 k-heads x 128 + 48 v-heads x 128
        // for qkv, 48 x 128 for z, 48 for a/b, hidden 2560), the shape the
        // decode profile's gdn.inproj number comes from.
        let qwen38 = ProcessInfo.processInfo.environment["NVMAI_BENCH_GDN_SHAPE"] == "qwen38"
        let qkvRows: UInt32 = qwen38 ? 16384 : 8192
        let zRows: UInt32 = qwen38 ? 6144 : 4096
        let abRows: UInt32 = qwen38 ? 48 : 32
        let N: UInt32 = qwen38 ? 2560 : 2048
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
        case "gdn_inproj_u4":
            kernelName2 = "gdn_in_proj_gemv_simd_u4"
            rowsPerTG = 8
        case "gdn_inproj_sk4":
            kernelName2 = "gdn_in_proj_gemv_simd_sk4"
            rowsPerTG = 2
        case "gdn_inproj_bw":
            kernelName2 = "gdn_in_proj_gemv_simd_bw"
            rowsPerTG = 8
        default:
            kernelName2 = "gdn_in_proj_gemv_simd"
            rowsPerTG = 8
        }
        // sk4 runs four simdgroups per row, so its threadgroup is wider than
        // rows * 32.
        let threadsPerTG = kernelName == "gdn_inproj_sk4" ? 256 : rowsPerTG * 32
        let pso = try context.pipeline(
            kernelName2, constants: [],
            maxTotalThreadsPerThreadgroup: threadsPerTG)

        // Full GDN decode chain (gdn_chain): in_proj + conv + qk_norm +
        // delta-step + gated_norm in one command buffer, mirroring the
        // decode's linear-attention block. Measures the extras' cost.
        let chain = kernelName == "gdn_chain"
        let convPSO = try context.pipeline("gdn_conv_mix_decode")
        let qkNormPSO = try context.pipeline(
            "gdn_qk_norm",
            constants: [MetalFunctionConstant(index: 95, value: .uint32(128))])
        let deltaPSO = try context.pipeline("gdn_delta_step_decode")
        let gatedNormPSO = try context.pipeline(
            "gdn_gated_norm",
            constants: [MetalFunctionConstant(index: 95, value: .uint32(128))])
        let convTail = makeBuffer(3 * Int(qkvRows), 0x44)          // [K-1, qkvDim] halfs
        let convW = makeBuffer(Int(qkvRows) * 4 * 2, 0x55)         // [qkvDim, K] bfloat
        let convOut = makeBuffer(Int(qkvRows) * 2, 0)
        let aLog = makeBuffer(Int(abRows) * 2, 0x60)
        let dtBias = makeBuffer(Int(abRows) * 2, 0x60)
        let state = makeBuffer(Int(abRows) * 128 * 128 * 4, 0)     // FP32 [Hv, Dv, Dk]
        let deltaY = makeBuffer(Int(zRows) * 2, 0)
        let gatedW = makeBuffer(Int(zRows) * 2, 0x66)
        let gatedOut = makeBuffer(Int(zRows) * 2, 0)

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
            enc.setComputePipelineState(pso)
            enc.dispatchThreadgroups(
                MTLSize(width: (totalRows + rowsPerTG - 1) / rowsPerTG, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
            if chain {
                // conv: tail(0) qkv(1) convW(2) out(3) channels(4) taps(5)
                enc.setComputePipelineState(convPSO)
                enc.setBuffer(convTail, offset: 0, index: 0)
                enc.setBuffer(qkvY, offset: 0, index: 1)
                enc.setBuffer(convW, offset: 0, index: 2)
                enc.setBuffer(convOut, offset: 0, index: 3)
                var ch = qkvRows
                var taps: UInt32 = 4
                enc.setBytes(&ch, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreads(MTLSize(width: Int(ch), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // qk_norm: convOut(0) kHeads(1) keyDim(2) rowStride(3)
                enc.setComputePipelineState(qkNormPSO)
                enc.setBuffer(convOut, offset: 0, index: 0)
                var kHeads: UInt32 = 16
                var keyDim: UInt32 = 128
                var rowStride = qkvRows
                enc.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 1)
                enc.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 2)
                enc.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: 2 * 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                // delta: convOut(0) aProj(1) bProj(2) aLog(3) dtBias(4) state(5) y(6) dims 7-10
                enc.setComputePipelineState(deltaPSO)
                enc.setBuffer(convOut, offset: 0, index: 0)
                enc.setBuffer(aY, offset: 0, index: 1)
                enc.setBuffer(bY, offset: 0, index: 2)
                enc.setBuffer(aLog, offset: 0, index: 3)
                enc.setBuffer(dtBias, offset: 0, index: 4)
                enc.setBuffer(state, offset: 0, index: 5)
                enc.setBuffer(deltaY, offset: 0, index: 6)
                var vHeads: UInt32 = 32
                var vDim: UInt32 = 128
                enc.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 7)
                enc.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 8)
                enc.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 9)
                enc.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 10)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(vHeads), height: Int(vDim) / 4, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
                // gated_norm: y(0) z(1) weight(2) out(3) vHeads(4) valueDim(5)
                enc.setComputePipelineState(gatedNormPSO)
                enc.setBuffer(deltaY, offset: 0, index: 0)
                enc.setBuffer(zY, offset: 0, index: 1)
                enc.setBuffer(gatedW, offset: 0, index: 2)
                enc.setBuffer(gatedOut, offset: 0, index: 3)
                enc.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(vHeads), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
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
        let gbPerSec = Double(bytes) / perIteration / 1_000_000_000
        let theoretical = 100.0
        print("kernel=\(kernelName) iterations=\(iterations) "
            + "total=\(String(format: "%.4f", totalSeconds))s "
            + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print("bytes/launch=\(bytes) "
            + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
            + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak")
    }

    /// The vocabulary head GEMV at the 35B family's shape (248,320 x 2048),
    /// the one GEMV every install runs at 8-bit. head_affine8 is the shipped
    /// generic affine kernel at bits=8 (measured 89.9 GB/s, at the ceiling;
    /// an 8-byte-per-lane specialization measured 91.7, noise, and was not
    /// kept), head_affine4 the same kernel at bits=4 (51.8 GB/s), head_int4
    /// the int4 head kernel (91.5). NVMAI_BENCH_HEAD_SHAPE=qwen38 takes
    /// 248,320 x 2560.
    private static func runHead(kernelName: String,
                                iterations: Int,
                                context: MetalContext) throws {
        let device = context.device
        let qwen38 = ProcessInfo.processInfo.environment["NVMAI_BENCH_HEAD_SHAPE"] == "qwen38"
        let rows: UInt32 = 248_320
        let n: UInt32 = qwen38 ? 2560 : 2048
        let groupCount = Int(n) / 64
        let bits: Int
        let kernel: String
        switch kernelName {
        case "head_affine4": bits = 4; kernel = "affine_quant_gemv_simd"
        case "head_int4": bits = 4; kernel = "dequant_int4_gemv_simd"
        default: bits = 8; kernel = "affine_quant_gemv_simd"
        }
        let rowBytes = Int(n) * bits / 8
        func makeBuffer(_ bytes: Int, _ value: UInt8) -> MTLBuffer {
            let buf = device.makeBuffer(length: bytes, options: .storageModeShared)!
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let w = makeBuffer(Int(rows) * rowBytes, 0x5A)
        let s = makeBuffer(Int(rows) * groupCount * 2, 0x3C)
        let bb = makeBuffer(Int(rows) * groupCount * 2, 0x00)
        let x = device.makeBuffer(length: Int(n) * 2, options: .storageModeShared)!
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(n) { xPtr[i] = Float16(Float(i % 97 + 1) * 0.001).bitPattern }
        let y = makeBuffer(Int(rows) * 2, 0)
        let constants = kernel.hasPrefix("affine")
            ? [MetalFunctionConstant(index: 100, value: .uint32(UInt32(bits)))] : []
        let pso = try context.pipeline(kernel, constants: constants,
                                       maxTotalThreadsPerThreadgroup: 256)
        var rowsVar = rows
        var nVar = n
        let threadgroups = (Int(rows) + 7) / 8
        let cb = context.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(s, offset: 0, index: 1)
        enc.setBuffer(bb, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        enc.setBytes(&rowsVar, length: 4, index: 5)
        enc.setBytes(&nVar, length: 4, index: 6)
        for _ in 0..<iterations {
            enc.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let total = cb.gpuEndTime - cb.gpuStartTime
        let per = total / Double(iterations)
        let bytes = UInt64(rows) * UInt64(rowBytes + groupCount * 4)
        let yPtr = y.contents().assumingMemoryBound(to: UInt16.self)
        let y0 = Float(Float16(bitPattern: yPtr[0]))
        let yLast = Float(Float16(bitPattern: yPtr[Int(rows) - 1]))
        print("kernel=\(kernel) bits=\(bits) n=\(n) iterations=\(iterations) "
            + "per_launch=\(String(format: "%.1f", per * 1_000_000))us "
            + "achieved=\(String(format: "%.1f", Double(bytes) / per / 1e9)) GB/s "
            + "y0=\(y0) yLast=\(yLast)")
    }

    /// What the CPU side-engine can actually read, and at how many threads.
    ///
    /// The premise of the side-engine is that NVMAI leaves the CPU idle --
    /// measured, 0.20 of one core out of eight while a 35B generates. That
    /// is true of the *cores* and says nothing about the *memory*, which is
    /// the resource decode is bound by on both sides. The GPU already reads
    /// at 74-88 GB/s during a 35B decode, near this machine's practical
    /// ceiling, and a 2B model reads about 1.9 GB per token at 8-bit. So the
    /// question this answers is not "is there a spare core" but "is there
    /// spare bandwidth, and what does taking it cost the model the person is
    /// waiting for".
    ///
    /// Run it twice -- idle, and with a generation in flight -- and the
    /// difference is the answer.
    ///
    ///     NVMAIBench cpu            # 8-bit, thread sweep
    static func runCPUGEMV(iterations: Int) {
        // Big enough that nothing is served from cache: the point is the
        // memory system, and a matrix that fits in the SLC measures the SLC.
        let rows = 8192
        let n = 8192
        let group = 64
        let weightBytes = rows * n
        let groups = rows * (n / group)
        print("cpu int8 affine gemv: \(rows)x\(n), "
              + "\(Double(weightBytes) / 1e6) MB of weights per pass, "
              + "\(iterations) passes")
        print("performance cores reported: \(Int8AffineGEMV.preferredThreads)")

        let weights = UnsafeMutablePointer<UInt8>.allocate(capacity: weightBytes)
        let scales = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let biases = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let x = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let out = UnsafeMutablePointer<Float>.allocate(capacity: rows)
        defer {
            weights.deallocate(); scales.deallocate(); biases.deallocate()
            x.deallocate(); out.deallocate()
        }
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> UInt64 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return state
        }
        for i in 0..<weightBytes { weights[i] = UInt8(truncatingIfNeeded: next()) }
        // 1.0 and 0.0 as BF16 bit patterns: the arithmetic is the same
        // whatever the constants, and the measurement is of the reads.
        for i in 0..<groups { scales[i] = 0x3F80; biases[i] = 0 }
        for i in 0..<n { x[i] = Float(i % 7) * 0.125 }

        let perPass = Double(weightBytes + groups * 4)
        print("  \("threads".padding(toLength: 8, withPad: " ", startingAt: 0))"
              + "\("ms/pass".padding(toLength: 10, withPad: " ", startingAt: 0))"
              + "\("GB/s".padding(toLength: 9, withPad: " ", startingAt: 0))"
              + "2B tok/s at 8-bit")
        for threads in [1, 2, 4, 6, 8] {
            // One untimed pass so the first one's page faults are not the
            // measurement.
            Int8AffineGEMV.threaded(weights: weights, scales: scales, biases: biases,
                                    x: x, rows: rows, n: n, out: out, threads: threads)
            let started = ContinuousClock.now
            for _ in 0..<iterations {
                Int8AffineGEMV.threaded(weights: weights, scales: scales, biases: biases,
                                        x: x, rows: rows, n: n, out: out, threads: threads)
            }
            let elapsed = started.duration(to: .now)
            let seconds = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            let perIteration = seconds / Double(iterations)
            let bandwidth = perPass / perIteration / 1e9
            // A 2B at 8-bit reads about 1.9 GB per token, the tied output
            // head included -- it is read in full for every token.
            let tokens = bandwidth / 1.9
            print(String(format: "  %-8d%-10.2f%-9.1f%.1f",
                         threads, perIteration * 1e3, bandwidth, tokens))
        }
        print("  (checksum \(out[0]))")
    }


    /// Hold the memory system at the side-engine's working width for a while.
    ///
    /// The companion to `runCPUGEMV`: that one asks what the CPU can read,
    /// this one exists so the same question can be asked of the GPU while
    /// the CPU is reading. A side-engine that halves the throughput of the
    /// model the person is waiting for is not a side-engine.
    static func runCPULoad(seconds: Double, threads: Int) {
        let rows = 8192, n = 8192, group = 64
        let weightBytes = rows * n, groups = rows * (n / group)
        let weights = UnsafeMutablePointer<UInt8>.allocate(capacity: weightBytes)
        let scales = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let biases = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let x = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let out = UnsafeMutablePointer<Float>.allocate(capacity: rows)
        defer {
            weights.deallocate(); scales.deallocate(); biases.deallocate()
            x.deallocate(); out.deallocate()
        }
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        for i in 0..<weightBytes {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            weights[i] = UInt8(truncatingIfNeeded: state)
        }
        for i in 0..<groups { scales[i] = 0x3F80; biases[i] = 0 }
        for i in 0..<n { x[i] = Float(i % 7) * 0.125 }

        print("cpu load: \(threads) threads for \(seconds)s")
        let deadline = ContinuousClock.now.advanced(by: .seconds(seconds))
        var passes = 0
        let started = ContinuousClock.now
        while ContinuousClock.now < deadline {
            Int8AffineGEMV.threaded(weights: weights, scales: scales, biases: biases,
                                    x: x, rows: rows, n: n, out: out, threads: threads)
            passes += 1
        }
        let elapsed = started.duration(to: .now)
        let taken = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        let bandwidth = Double(passes) * Double(weightBytes + groups * 4) / taken / 1e9
        print(String(format: "  %d passes in %.1fs, %.1f GB/s (checksum %.0f)",
                     passes, taken, bandwidth, out[0]))
    }


    /// Qwen3.5-2B on the CPU, checked against the continuations that define
    /// correctness for the numpy reference.
    ///
    /// The token ids come out of the snapshot's own `vocab.json`, so this
    /// needs no tokenizer and cannot drift from what the reference does.
    static func runCPUQwen35(snapshot path: String, dump: URL? = nil) throws {
        let directory = URL(fileURLWithPath: path)
        let started = ContinuousClock.now
        let snapshot = try AffineSnapshot(directory: directory)
        // Width is the side-engine's scheduling knob, so it is settable
        // here: the measurement that produced the policy is a sweep of it.
        let requested = ProcessInfo.processInfo.environment["NVMAI_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        func seconds(_ from: ContinuousClock.Instant) -> Double {
            let elapsed = from.duration(to: .now)
            return Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
        }
        print("\(path): \(snapshot.configuration.layers) layers, "
              + "hidden \(snapshot.configuration.hiddenSize), "
              + "rotary \(snapshot.configuration.rotaryDim)/"
              + "\(snapshot.configuration.headDim), "
              + "loaded in \(String(format: "%.2fs", seconds(started)))")
        print("threads: \(model.threads)")

        let vocabulary = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        guard let vocabulary = vocabulary as? [String: Int] else {
            print("vocab.json is not a mapping"); return
        }
        var inverse: [Int: String] = [:]
        inverse.reserveCapacity(vocabulary.count)
        for (text, id) in vocabulary { inverse[id] = text }

        let checks: [([String], String)] = [
            (["Once", "\u{120}upon", "\u{120}a"], "\u{120}time"),
            (["The", "\u{120}capital", "\u{120}of", "\u{120}France", "\u{120}is"],
             "\u{120}Paris"),
            (["The", "\u{120}quick", "\u{120}brown", "\u{120}fox", "\u{120}jumps",
              "\u{120}over", "\u{120}the", "\u{120}lazy"], "\u{120}dog"),
        ]
        var failures = 0
        for (index, (words, expected)) in checks.enumerated() {
            model.reset()
            var logits: [Float] = []
            let run = ContinuousClock.now
            for word in words {
                guard let id = vocabulary[word] else {
                    print("  no token for \(word)"); failures += 1; break
                }
                logits = try model.step(token: id)
            }
            guard !logits.isEmpty else { continue }
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            let want = vocabulary[expected] ?? -1
            let ok = best == want
            failures += ok ? 0 : 1
            let prompt = words.map { $0.replacingOccurrences(of: "\u{120}", with: " ") }
                .joined()
            let rate = Double(words.count) / seconds(run)
            print(String(format: "  %@ %-46@ -> %@ (%.2f), wanted %@  [%.1f tok/s]",
                         ok ? "ok " : "FAIL", prompt as NSString,
                         inverse[best] ?? "?", logits[best], expected, rate))
            if let dump {
                try? FileManager.default.createDirectory(
                    at: dump, withIntermediateDirectories: true)
                let file = dump.appendingPathComponent("check\(index).f32")
                let payload = logits.withUnsafeBufferPointer { Data(buffer: $0) }
                try? payload.write(to: file)
            }
        }
        print(failures == 0 ? "all continuations correct"
              : "\(failures) of \(checks.count) wrong")
    }


    /// The CPU side-engine's commands. Returns whether one ran, so `main`
    /// can dispatch them before it creates a Metal context.
    static func runCPUCommand(_ name: String, iterations: Int) throws -> Bool {
        switch name {
        case "cpuload":
            // Sustained load at one width, for measuring what the side-engine
            // costs the model the person is waiting for. `iterations` is
            // seconds here; the third argument is the thread count.
            let threads = CommandLine.arguments.count > 3
                ? Int(CommandLine.arguments[3]) ?? 4 : 4
            runCPULoad(seconds: Double(iterations), threads: threads)
        case "cpu35":
            // The side-engine's model, on the same continuations the numpy
            // reference checks itself with. Agreement here is what says the
            // Swift forward pass matches the oracle.
            let snapshot = CommandLine.arguments.count > 2
                ? CommandLine.arguments[2] : ".build/qwen35-2b-affine-8bit"
            // An optional directory to write each check's full logit vector
            // into, so parity is a number rather than an impression.
            let dump = CommandLine.arguments.count > 3
                ? URL(fileURLWithPath: CommandLine.arguments[3]) : nil
            try runCPUQwen35(snapshot: snapshot, dump: dump)
        case let other where other.hasPrefix("cpu"):
            runCPUGEMV(iterations: iterations)
        default:
            return false
        }
        return true
    }

}
