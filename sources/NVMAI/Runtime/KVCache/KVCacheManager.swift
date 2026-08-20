import Foundation
import Darwin
import Metal

/// Which attention variant a layer runs. Qwen 3.6 interleaves 30
/// gated-DeltaNet linear-attention layers with 10 full-attention layers;
/// linear layers keep a fixed-size recurrent state (owned by
/// `GDNStateManager`) instead of per-token K/V rows. Sourced from
/// `ArchConfig.fullAttentionLayerMask` (0 = swa, 1 = full, 2 = linear).
public enum LayerKind: Sendable { case swa, full, linear }

/// A read view the attention kernels bind. `offset` stays 0; ring-enabled SWA
/// layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
/// unchecked-invariant: every stored property is a `let`. The type is only
/// @unchecked because MTLBuffer is not Sendable; the struct itself is a
/// read-only descriptor of a range, and callers that write through it are
/// serialised by whoever owns the buffer.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Always 0 under linear storage.
    public let offset: Int
    /// Bytes per token, including affine metadata for quantized storage.
    public let stride: Int
    /// Number of valid positions written so far (== `position`). Attention reads
    /// `[0, validTokenCount]` (inclusive of the just-written token).
    public let validTokenCount: Int
    public let precision: KVCachePrecision
    public let valueBytes: Int
    public let groupSize: Int
}

/// Per-layer K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool
    public let precision: KVCachePrecision
    /// Retained so `reserve` can allocate; init-only allocation was the previous
    /// invariant and growth deliberately relaxes it.
    private let device: MTLDevice

    private var kBuffers: [MTLBuffer]
    private var vBuffers: [MTLBuffer]
    private let strides:  [Int]         // bytes per token, per layer
    private let kinds:    [LayerKind]
    private var capacityTokens: [Int]
    private let valueBytes: [Int]

    public private(set) var position: Int = 0

    private static let fp16Size = 2
    public static let quantizationGroupSize = 64

    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                fp16RingEnabled: Bool = false,
                precision: KVCachePrecision = .fp16,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        self.device = device
        self.config = config
        self.maxContext = maxContext
        self.precision = precision
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, (slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens))

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        var valueByteCounts: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)
        valueByteCounts.reserveCapacity(config.numLayers)

        // Linear-attention layers keep no per-token K/V rows; they share one
        // page-sized placeholder so the parallel arrays stay non-optional.
        var linearPlaceholder: MTLBuffer? = nil

        for layer in 0..<config.numLayers {
            let maskValue = config.fullAttentionLayerMask[layer]
            if maskValue == 2 {
                let placeholder: MTLBuffer
                if let existing = linearPlaceholder {
                    placeholder = existing
                } else {
                    guard let made = device.makeBuffer(length: Int(getpagesize()),
                                                       options: .storageModeShared) else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    made.label = "kv.linear-placeholder"
                    linearPlaceholder = made
                    placeholder = made
                }
                ks.append(placeholder)
                vs.append(placeholder)
                st.append(0)
                kd.append(.linear)
                caps.append(0)
                valueByteCounts.append(0)
                continue
            }
            let isFull = maskValue != 0
            let fp16Stride = isFull ? fullStride : swaStride
            let elements = fp16Stride / Self.fp16Size
            let layout = Self.rowLayout(elements: elements, precision: precision)
            let stride = layout.stride
            // Linear layers start at `initialCapacityTokens` and grow on demand
            // rather than reserving `maxContext` up front. At 262144 tokens a
            // full reservation is 512 MiB per layer, 20 GiB across 40 -- lazily
            // touched, so it barely shows in RSS, but on a 24 GB machine the
            // mappings alone cost throughput: the same 25-token prompt measured
            // 13.80 s of decode at maxContext 8192 against 22.44 s at 262144.
            // Growing keeps a short conversation at a short conversation's cost
            // while leaving the advertised limit reachable.
            let capacity = ringEnabled && !isFull
                ? swaCapacity
                : min(maxContext, Self.initialCapacityTokens)
            let length = capacity * stride

            guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            kBuf.label = "kv.K.layer\(layer)"
            ks.append(kBuf)

            guard let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            vBuf.label = "kv.V.layer\(layer)"
            vs.append(vBuf)

            st.append(stride)
            kd.append(isFull ? .full : .swa)
            caps.append(capacity)
            valueByteCounts.append(layout.valueBytes)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
        self.valueBytes = valueByteCounts
    }

    /// Tokens each linear layer is sized for before any growth.
    ///
    /// 8192 because it measured as fast as any smaller reservation and holds an
    /// ordinary conversation without a single grow. Capacity doubles from here.
    public static let initialCapacityTokens = 8_192

    /// Grows linear layers so every one can hold `tokens`, copying what is
    /// already stored.
    ///
    /// Must be called before writing at a position beyond the current capacity.
    /// Ring-backed SWA layers are never grown -- their capacity is the window and
    /// is deliberate. Capacity doubles, so a conversation reaching the advertised
    /// 262144 limit pays five copies in total rather than one per token.
    ///
    /// Buffers are `storageModeShared`, so the copy is a plain `memcpy`; callers
    /// fetch buffers through the accessors at use time and never cache them
    /// across tokens, which is what makes swapping them safe here.
    public func reserve(tokens: Int) throws {
        let needed = min(max(tokens, 1), maxContext)
        for layer in 0..<kinds.count {
            guard kinds[layer] != .linear else { continue }
            if fp16RingEnabled && kinds[layer] == .swa { continue }
            let current = capacityTokens[layer]
            guard current < needed else { continue }
            var target = current
            while target < needed { target *= 2 }
            target = min(target, maxContext)

            let stride = strides[layer]
            let length = target * stride
            guard let newK = device.makeBuffer(length: length, options: .storageModeShared),
                  let newV = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            newK.label = "kv.K.layer\(layer)"
            newV.label = "kv.V.layer\(layer)"
            let usedBytes = min(current, position) * stride
            if usedBytes > 0 {
                memcpy(newK.contents(), kBuffers[layer].contents(), usedBytes)
                memcpy(newV.contents(), vBuffers[layer].contents(), usedBytes)
            }
            kBuffers[layer] = newK
            vBuffers[layer] = newV
            capacityTokens[layer] = target
        }
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// Bytes per token for `layer` (K and V share the same stride).
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// Total bytes of the K buffer for `layer`.
    public func bufferLength(layer: Int) -> Int {
        return capacityTokens[layer] * strides[layer]
    }

    /// Write target for this layer's K projection at `position`.
    public func kSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear, "linear layers have no KV slots")
        validateRange(start: position, count: 1)
        return (kBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`. Always
    /// distinct from `kSlot`: K runs per-head k_norm + RoPE while V is
    /// normed differently, so the two buffers cannot alias.
    public func vSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear, "linear layers have no KV slots")
        validateRange(start: position, count: 1)
        return (vBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    public func kRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (kBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (vBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, validTokenCount: position)
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return makeView(buffer: kBuffers[layer], layer: layer,
                        offset: 0, validTokenCount: validTokenCount)
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, validTokenCount: position)
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return makeView(buffer: vBuffers[layer], layer: layer,
                        offset: 0, validTokenCount: validTokenCount)
    }

    public func keyRangeView(layer: Int, start: Int, count: Int) -> KVView {
        let range = kRange(layer: layer, start: start, count: count)
        return makeView(buffer: range.buffer, layer: layer, offset: range.offset,
                        validTokenCount: count)
    }

    public func valueRangeView(layer: Int, start: Int, count: Int) -> KVView {
        let range = vRange(layer: layer, start: start, count: count)
        return makeView(buffer: range.buffer, layer: layer, offset: range.offset,
                        validTokenCount: count)
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
    }

    /// Rewind the logical cursor without copying KV. Speculative rows are
    /// append-only and become unreachable immediately; a later pass
    /// overwrites them.
    /// Ring-backed draft KV is safe because MTP verification never rewinds by
    /// more than its two-token proposal depth.
    func rewind(to newPosition: Int) throws {
        guard newPosition >= 0, newPosition <= position else {
            throw InferenceStateSnapshotError.invalidPosition(newPosition)
        }
        position = newPosition
    }

    /// Drop all cached positions and return physical pages to the OS.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. `MADV_DONTNEED` on the page-aligned span
    /// releases resident memory between turns; pages fault back in on next write.
    public func reset() {
        position = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
            advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
        }
    }

    func snapshotSegmentLengths(at snapshotPosition: Int) throws -> [Int] {
        guard snapshotPosition > 0, snapshotPosition <= maxContext else {
            throw InferenceStateSnapshotError.invalidPosition(snapshotPosition)
        }
        var lengths: [Int] = []
        lengths.reserveCapacity(config.numLayers * 2)
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let storedTokens = min(snapshotPosition, capacityTokens[layer])
            let (length, overflow) = storedTokens.multipliedReportingOverflow(
                by: strides[layer])
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            lengths.append(length)
            lengths.append(length)
        }
        return lengths
    }

    func appendSnapshotPayload(to payload: inout Data,
                               segmentLengths: [Int]) throws {
        let expected = try snapshotSegmentLengths(at: position)
        guard segmentLengths == expected else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        var segment = 0
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let kLength = segmentLengths[segment]
            payload.append(kBuffers[layer].contents().assumingMemoryBound(to: UInt8.self),
                           count: kLength)
            segment += 1
            let vLength = segmentLengths[segment]
            payload.append(vBuffers[layer].contents().assumingMemoryBound(to: UInt8.self),
                           count: vLength)
            segment += 1
        }
    }

    func restoreSnapshot(position snapshotPosition: Int,
                         segmentLengths: [Int],
                         bytes: UnsafeRawBufferPointer,
                         offset: inout Int) throws {
        let expected = try snapshotSegmentLengths(at: snapshotPosition)
        guard segmentLengths == expected else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        reset()
        var segment = 0
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let kLength = segmentLengths[segment]
            try copySnapshotSegment(bytes: bytes,
                                    offset: &offset,
                                    length: kLength,
                                    destination: kBuffers[layer])
            segment += 1
            let vLength = segmentLengths[segment]
            try copySnapshotSegment(bytes: bytes,
                                    offset: &offset,
                                    length: vLength,
                                    destination: vBuffers[layer])
            segment += 1
        }
        position = snapshotPosition
    }

    private func copySnapshotSegment(bytes: UnsafeRawBufferPointer,
                                     offset: inout Int,
                                     length: Int,
                                     destination: MTLBuffer) throws {
        guard length <= destination.length,
              offset >= 0,
              length >= 0,
              offset <= bytes.count - length,
              let source = bytes.baseAddress?.advanced(by: offset) else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        memcpy(destination.contents(), source, length)
        offset += length
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func makeView(buffer: MTLBuffer, layer: Int, offset: Int,
                          validTokenCount: Int) -> KVView {
        KVView(buffer: buffer, offset: offset, stride: strides[layer],
               validTokenCount: validTokenCount, precision: precision,
               valueBytes: valueBytes[layer], groupSize: Self.quantizationGroupSize)
    }

    private static func rowLayout(elements: Int,
                                  precision: KVCachePrecision) -> (stride: Int, valueBytes: Int) {
        if precision == .fp16 {
            return (elements * fp16Size, elements * fp16Size)
        }
        let packed = (elements * precision.rawValue + 7) / 8
        let alignedPacked = (packed + 1) & ~1
        let groups = (elements + quantizationGroupSize - 1) / quantizationGroupSize
        return (alignedPacked + groups * 2 * fp16Size, alignedPacked)
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        precondition(capacityTokens[layer] > 0, "layer has no KV storage")
        return position % capacityTokens[layer]
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
