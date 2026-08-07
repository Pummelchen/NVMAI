import Foundation
import Metal
import NVMAI

/// Kernel micro-benchmark: measures the fused QKV GEMV's achieved memory
/// bandwidth in isolation (synthetic buffers, no model pipeline), so kernel
/// variants can be compared directly. Reports GB/s against the M3's
/// theoretical peak (~100 GB/s) — the "42% of peak" attention-block number
/// reproduced here means the limiter is in this kernel, not the pipeline.
///
/// Usage: NVMAIBench [kernelName] [iterations]
///   kernelName: baseline (default), bandwidth, unroll2, ulong2
@main
struct NVMAIBench {
    static func main() throws {
        let kernelName = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1] : "baseline"
        let iterations = CommandLine.arguments.count > 2
            ? Int(CommandLine.arguments[2]) ?? 300 : 300

        let context = try MetalContext()
        let device = context.device
        print("device: \(device.name)")

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
}
