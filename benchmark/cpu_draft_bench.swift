import Foundation
import Accelerate

/// Parallel fused int4-GEMV: dequantize in registers and dot with x in one
/// pass over the int4 rows (no FP16 staging), spread over the 8 cores.
/// This is the realistic CPU draft shape — the LM head is 262144 x 2048
/// int4 (268 MiB), the dominant read per draft token.
let D = 2048
let vocab = 262_144
let groupSize = 64

func makeInt4Buffer(_ bytes: Int) -> [UInt8] {
    var buf = [UInt8](repeating: 0, count: bytes)
    for i in 0..<bytes { buf[i] = UInt8((i &* 2654435761) & 0xFF) }
    return buf
}

func makeScales(_ count: Int) -> [Float16] {
    var s = [Float16](repeating: 0, count: count)
    for i in 0..<count { s[i] = Float16(Float(i % 7) * 0.01 + 0.1) }
    return s
}

func makeX(_ n: Int) -> [Float16] {
    var x = [Float16](repeating: 0, count: n)
    for i in 0..<n { x[i] = Float16(Float(i % 13) * 0.001 + 0.5) }
    return x
}

/// One fused row GEMV: y[m] = sum_n dequant(W[m,n]) * x[n].
func fusedRow(_ rowBase: UnsafeRawPointer, _ scaleBase: UnsafePointer<Float16>,
              _ x: UnsafePointer<Float16>, _ row: Int) -> Float {
    let bytesPerRow = D / 2
    let rowPtr = rowBase.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
    let groupsPerRow = D / groupSize
    let scales = scaleBase.advanced(by: row * groupsPerRow)
    var acc: Float = 0
    var g = 0
    var byteOff = 0
    while g < groupsPerRow {
        let sc = Float(scales[g])
        var j = 0
        while j < 32 {                    // 32 bytes of int4 = 64 elements
            let b = rowPtr[byteOff + j]
            acc += Float(b & 0x0F) * sc * Float(x[g * 64 + j * 2])
            acc += Float(b >> 4) * sc * Float(x[g * 64 + j * 2 + 1])
            j += 1
        }
        byteOff += 32
        g += 1
    }
    return acc
}

let headBytes = vocab * D / 2
let w = makeInt4Buffer(headBytes)
let scales = makeScales(vocab * D / groupSize)
let x = makeX(D)
var y = [Float](repeating: 0, count: vocab)
print("head: \(headBytes/1_048_576) MiB int4, \(vocab) rows")

// warmup
let wp = w.withUnsafeBytes { $0.baseAddress! }
let sp = scales.withUnsafeBufferPointer { $0.baseAddress! }
let xp = x.withUnsafeBufferPointer { $0.baseAddress! }
_ = fusedRow(wp, sp, xp, 0)

let t0 = DispatchTime.now().uptimeNanoseconds
DispatchQueue.concurrentPerform(iterations: vocab) { row in
    let v = fusedRow(wp, sp, xp, row)
    y[row] = v
}
let t1 = DispatchTime.now().uptimeNanoseconds
let gbs = Double(headBytes) / Double(t1 - t0) * 1e9 / 1e9
let perPass = Double(t1 - t0) / 1e6
let sum = y.prefix(64).reduce(0, +)
print(String(format: "fused parallel GEMV: %.1f GB/s, %.2f ms/pass (sum %.1f)", gbs, perPass, sum))
print(String(format: "K=4: %.1f ms | K=8: %.1f ms | GPU window: 50 ms", perPass * 4, perPass * 8))
