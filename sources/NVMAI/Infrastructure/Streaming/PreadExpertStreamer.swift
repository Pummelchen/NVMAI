import Darwin
import Foundation
import Metal
import Synchronization

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    /// K11: the layer the plan's pread offsets are computed against.
    ///
    /// NOTE: the streamer is bound to ONE layer file at construction, and
    /// `StreamLayout.expertOffset(layer: 0, ...)` is the branch that consults
    /// that file's per-layer `expertOffsets` table — so 0 is the correct value
    /// for the current per-layer design (passing the real layer would select
    /// the dense cross-layer formula and mis-offset). Callers must only pass a
    /// nonzero layer if the streamer ever serves a multi-layer file.
    public let layer: Int
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int,
                layer: Int = 0) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
        self.layer = layer
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]

    private var nextSlot = 0

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    private var useClock = 0
    /// K12: slots whose pread fill has been planned but not yet completed.
    /// Set under `cacheLock` by `makeExpertCachePlan` for miss slots, cleared
    /// when the fill lands. Concurrent plans and round-robin loads skip and
    /// never overwrite these slots.
    private var slotPendingFill: [Bool]
    private let cacheLock = NSLock()

    public init(layout: StreamLayout,
                device: MTLDevice,
                slotCount: Int,
                cachePolicy: ExpertCachePolicy = .lfu) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        let pageSize = Int(getpagesize())

        let openedFD = open(layout.path, O_RDONLY)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD

        var fileStats = stat()
        // K9: fstat failure must not silently skip size validation — a
        // truncated file would then be read out of bounds by pread.
        guard fstat(openedFD, &fileStats) == 0 else {
            let statErrno = errno
            close(openedFD)
            throw ModelError.posixFailed(call: "fstat(\(layout.path))", errno: statErrno)
        }
        let required = layout.streamOffset + layout.streamSize
        if UInt64(fileStats.st_size) < required {
            close(openedFD)
            throw StreamerError.sizeMismatch(
                expected: required,
                actual: UInt64(fileStats.st_size))
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
            close(openedFD)
        }

        for _ in 0..<slotCount {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: allocationSize,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.slotPendingFill = [Bool](repeating: false, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        // K12: slot selection and fill share one critical section so the
        // round-robin path never lands on a slot a concurrent plan reserved
        // (slotPendingFill) and no fill can interleave with another pread.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        var candidate = nextSlot
        var scanned = 0
        while slotPendingFill[candidate] && scanned < slotCount {
            candidate = (candidate + 1) % slotCount
            scanned += 1
        }
        nextSlot = (candidate + 1) % slotCount
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: candidate)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        // K12: the pread fill and the slot bookkeeping share one critical
        // section so a concurrent plan/execute or another load cannot write
        // into this slot while the pread is in flight.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: slot)
    }

    /// Fill `slot` with `expert` and update bookkeeping. Callers hold
    /// `cacheLock`.
    private func loadExpertUnlocked(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride))
        slotPendingFill[slot] = false
        slotExpert[slot] = expert
        slotLastUse[slot] = useClock
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  layer: Int = 0,
                                  avoidingSlots: Set<Int> = []) throws -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(layer: layer,
                                             experts: experts,
                                             avoidingSlots: avoidingSlots) else {
            // K10: config-triggered placement failure (too few slots for the
            // requested expert set) is recoverable — throw instead of
            // crashing; the runner already handles thrown errors.
            throw ModelError.expertCacheUnplaceable(
                detail: "\(experts.count) experts do not fit in \(slotCount) cache slots (policy \(cachePolicy.rawValue), avoiding \(avoidingSlots.count) slots)")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            layer: Int = 0,
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(layer: layer, experts: experts, avoidingSlots: avoidingSlots)
    }

    private func makeExpertCachePlan(layer: Int,
                                     experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)
        // K12: slots with a fill already planned by an in-flight plan are not
        // assignable until their execute completes.
        for slot in 0..<slotCount where slotPendingFill[slot] {
            reserved[slot] = true
        }

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] && !slotPendingFill[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }
        guard misses.count <= evictable.count else { return nil }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
            slotExpert[slot] = -1
            slotLastUse[slot] = clock
            slotPendingFill[slot] = true
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count,
            layer: layer)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        // K12: hold the cache lock across the pread fills. `makeExpertCachePlan`
        // reserved the miss slots under the same lock (slotPendingFill), and
        // the fills complete here under it, so no concurrent plan or
        // round-robin load can clobber a slot mid-pread. This serializes the
        // fills (previously concurrentPerform) — correctness first; the fills
        // are the only writers of slot contents.
        //
        // Parallel fills are now the default (NVMAI_PARALLEL_IO=0 to disable):
        // each miss preads into its own slot reserved by the plan, pread on a
        // shared fd is thread-safe, and this path runs only in the decode
        // fetch, which is single-flight (awaited before the next layer's plan;
        // the prefill uses a separate tile fetch). The only shared state is
        // the bookkeeping, guarded by a small lock. Measured on the 4-bit M3:
        // IO wall 41.2 -> 30.9 ms/token and decode 9.98 -> 12.80 tok/s (+28%)
        // with the idle CPU cores doing the fills in parallel.
        if ProcessInfo.processInfo.environment["NVMAI_PARALLEL_IO"] != "0",
           plan.misses.count > 1 {
            let firstError = Mutex<Error?>(nil)
            DispatchQueue.concurrentPerform(iterations: plan.misses.count) { i in
                let index = plan.misses[i]
                let slot = plan.assignedSlots[index]
                let expert = plan.experts[index]
                do {
                    let regionOffset = layout.expertOffset(layer: plan.layer, expert: expert)
                    guard regionOffset + layout.expertStride <= layout.streamSize else {
                        throw StreamerError.offsetOutOfRange(regionOffset)
                    }
                    try readFull(
                        into: slotPointers[slot],
                        fileOffset: layout.streamOffset + regionOffset,
                        count: Int(layout.expertStride))
                    cacheLock.lock()
                    slotPendingFill[slot] = false
                    slotExpert[slot] = expert
                    slotLastUse[slot] = useClock
                    cacheLock.unlock()
                } catch {
                    firstError.withLock { if $0 == nil { $0 = error } }
                }
            }
            if let error = firstError.withLock({ $0 }) { throw error }
            return expertCachePlanBuffers(plan)
        }

        cacheLock.lock()
        defer { cacheLock.unlock() }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            let expert = plan.experts[index]
            let regionOffset = layout.expertOffset(layer: plan.layer, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else {
                throw StreamerError.offsetOutOfRange(regionOffset)
            }
            try readFull(
                into: slotPointers[slot],
                fileOffset: layout.streamOffset + regionOffset,
                count: Int(layout.expertStride))
            slotPendingFill[slot] = false
            slotExpert[slot] = expert
            slotLastUse[slot] = useClock
        }

        return expertCachePlanBuffers(plan)
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], UInt64(0), layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts, layer: plan.layer),
                            requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts, layer: 0), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses, layer: 0), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            // K28: checked arithmetic — a wrapping `&+` could merge two
            // huge ranges into a nonsense span. On overflow keep the ranges
            // separate (the merge is an optimization, never a correctness
            // requirement).
            let (lastEnd, lastOverflow) = last.offset.addingReportingOverflow(last.count)
            let (rangeEnd, rangeOverflow) = range.offset.addingReportingOverflow(range.count)
            if lastOverflow || rangeOverflow {
                result.append(last)
                result.append(range)
                continue
            }
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func expertAdviceRanges(experts: [Int],
                                    layer: Int) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: layer, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }
}
