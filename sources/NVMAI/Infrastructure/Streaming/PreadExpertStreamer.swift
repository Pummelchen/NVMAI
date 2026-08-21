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
    /// Slot incarnation captured when this plan reserved or hit each slot.
    /// A command may use the slot only while this generation still matches.
    public let assignedGenerations: [UInt64]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int],
                assignedGenerations: [UInt64], misses: [Int], hits: Int,
                layer: Int = 0) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.assignedGenerations = assignedGenerations
        self.misses = misses
        self.hits = hits
        self.layer = layer
    }
}

public struct ExpertStreamingStatistics: Sendable, Equatable {
    public let plans: UInt64
    public let requestedExperts: UInt64
    public let hits: UInt64
    public let misses: UInt64
    public let bytesRead: UInt64
    public let readOperations: UInt64
    public let evictions: UInt64
    public let reloads: UInt64
    public let loadBatches: UInt64
    public let totalLoadNanos: UInt64
    public let maximumLoadNanos: UInt64
    public let latencyHistogram: [UInt64]
    public let residentSlots: Int
    public let loadingSlots: Int
    public let pinnedSlots: Int
    public let peakLoadingSlots: Int

    public var hitRate: Double {
        requestedExperts == 0 ? 0 : Double(hits) / Double(requestedExperts)
    }

    /// An upper-bound estimate from a fixed, bounded power-of-two histogram.
    public func loadLatencyPercentile(_ percentile: Double) -> UInt64 {
        guard loadBatches > 0 else { return 0 }
        let clamped = min(1, max(0, percentile))
        let rank = max(UInt64(1), UInt64(ceil(clamped * Double(loadBatches))))
        var cumulative: UInt64 = 0
        for (index, count) in latencyHistogram.enumerated() {
            cumulative &+= count
            if cumulative >= rank {
                return PreadExpertStreamer.latencyBucketUpperBound(index: index)
            }
        }
        return UInt64.max
    }

    public static let zero = ExpertStreamingStatistics(
        plans: 0, requestedExperts: 0, hits: 0, misses: 0,
        bytesRead: 0, readOperations: 0, evictions: 0, reloads: 0,
        loadBatches: 0, totalLoadNanos: 0, maximumLoadNanos: 0,
        latencyHistogram: [UInt64](repeating: 0, count: 17),
        residentSlots: 0, loadingSlots: 0, pinnedSlots: 0, peakLoadingSlots: 0)

    func adding(_ other: ExpertStreamingStatistics) -> ExpertStreamingStatistics {
        ExpertStreamingStatistics(
            plans: plans &+ other.plans,
            requestedExperts: requestedExperts &+ other.requestedExperts,
            hits: hits &+ other.hits,
            misses: misses &+ other.misses,
            bytesRead: bytesRead &+ other.bytesRead,
            readOperations: readOperations &+ other.readOperations,
            evictions: evictions &+ other.evictions,
            reloads: reloads &+ other.reloads,
            loadBatches: loadBatches &+ other.loadBatches,
            totalLoadNanos: totalLoadNanos &+ other.totalLoadNanos,
            maximumLoadNanos: max(maximumLoadNanos, other.maximumLoadNanos),
            latencyHistogram: zip(latencyHistogram, other.latencyHistogram)
                .map { $0 &+ $1 },
            residentSlots: residentSlots + other.residentSlots,
            loadingSlots: loadingSlots + other.loadingSlots,
            pinnedSlots: pinnedSlots + other.pinnedSlots,
            peakLoadingSlots: max(peakLoadingSlots, other.peakLoadingSlots))
    }

    public func subtracting(_ baseline: ExpertStreamingStatistics) -> ExpertStreamingStatistics {
        func delta(_ value: UInt64, _ base: UInt64) -> UInt64 {
            value >= base ? value - base : 0
        }
        return ExpertStreamingStatistics(
            plans: delta(plans, baseline.plans),
            requestedExperts: delta(requestedExperts, baseline.requestedExperts),
            hits: delta(hits, baseline.hits),
            misses: delta(misses, baseline.misses),
            bytesRead: delta(bytesRead, baseline.bytesRead),
            readOperations: delta(readOperations, baseline.readOperations),
            evictions: delta(evictions, baseline.evictions),
            reloads: delta(reloads, baseline.reloads),
            loadBatches: delta(loadBatches, baseline.loadBatches),
            totalLoadNanos: delta(totalLoadNanos, baseline.totalLoadNanos),
            maximumLoadNanos: maximumLoadNanos,
            latencyHistogram: zip(latencyHistogram, baseline.latencyHistogram)
                .map { delta($0, $1) },
            residentSlots: residentSlots,
            loadingSlots: loadingSlots,
            pinnedSlots: pinnedSlots,
            peakLoadingSlots: peakLoadingSlots)
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
    case agingLFU = "aging-lfu"
}

public enum ExpertIOBackend: String, Sendable {
    case pread
    case metal

    static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ExpertIOBackend {
        guard let raw = environment["NVMAI_EXPERT_IO_BACKEND"] else { return .pread }
        guard let backend = ExpertIOBackend(rawValue: raw) else {
            throw ModelError.internalInconsistency(
                detail: "unsupported NVMAI_EXPERT_IO_BACKEND '\(raw)'; allowed: pread, metal")
        }
        return backend
    }
}

public enum ExpertCacheLayout: String, Sendable {
    case perSlot = "per-slot"
    case pool

    static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ExpertCacheLayout {
        guard let raw = environment["NVMAI_EXPERT_CACHE_LAYOUT"] else { return .perSlot }
        guard let layout = ExpertCacheLayout(rawValue: raw) else {
            throw ModelError.internalInconsistency(
                detail: "unsupported NVMAI_EXPERT_CACHE_LAYOUT '\(raw)'; allowed: per-slot, pool")
        }
        return layout
    }
}

/// SSD-backed routed-expert streamer with a fixed per-layer slot cache.
/// unchecked-invariant: the expert cache bookkeeping is guarded by `cacheLock`,
/// which is what lets `DispatchQueue.concurrentPerform` fan the misses out
/// across threads. Slot state is published only after all direct reads finish,
/// so concurrent planners never treat partial bytes as resident.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy
    public let ioBackend: ExpertIOBackend
    public let cacheLayout: ExpertCacheLayout
    public let poolSlotStride: Int

    private let fd: Int32

    /// Bounded-footprint reader. On by default; `NVMAI_BOUNDED_IO=0` opts out.
    ///
    /// Opens its own F_NOCACHE descriptors so expert reads never enter the unified
    /// buffer cache. That makes the slot budget the machine's true footprint,
    /// which is the whole point of streaming a 35B model on 24 GB.
    ///
    /// It is not free. Measured against the page-cache path it costs 15-30% of
    /// decode throughput, because every miss becomes a real device read instead of
    /// a cache hit -- and the cost is worst exactly where the hit rate is lowest
    /// (-40% at the 8-slot floor against -18% at 16 slots).
    ///
    /// It is the default anyway. The page-cache path is faster only by borrowing
    /// memory it never declares: process RSS looks smaller while the OS holds the
    /// difference, so "a 35B model in 1 GB" stops being true. A footprint you can
    /// account for is the product; throughput is what is being traded for it.
    private let boundedReader: ParallelExpertReader?
    private let metalReader: MetalExpertReader?
    private let eventCoordinator: ExpertIOEventCoordinator?
    private let metalStagingPool: MetalExpertStagingPool?
    private let metalIOService: MetalExpertIOService?
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]
    private let slotBufferOffsets: [UInt64]
    private let residencyTable: MTLBuffer

    private var nextSlot = 0

    private enum SlotState: UInt8 {
        case empty
        case loading
        case resident
    }

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var slotState: [SlotState]
    private var slotGeneration: [UInt64]
    private var slotPinCount: [Int]
    private var expertUseCount: [Int]
    private var expertLoadCount: [Int]
    private var useClock = 0
    private var statisticsPlans: UInt64 = 0
    private var statisticsRequestedExperts: UInt64 = 0
    private var statisticsHits: UInt64 = 0
    private var statisticsMisses: UInt64 = 0
    private var statisticsBytesRead: UInt64 = 0
    private var statisticsReadOperations: UInt64 = 0
    private var statisticsEvictions: UInt64 = 0
    private var statisticsReloads: UInt64 = 0
    private var statisticsLoadBatches: UInt64 = 0
    private var statisticsTotalLoadNanos: UInt64 = 0
    private var statisticsMaximumLoadNanos: UInt64 = 0
    private var statisticsLatencyHistogram = [UInt64](repeating: 0, count: 17)
    private var statisticsPeakLoadingSlots = 0
    private let cacheLock = NSLock()

    public init(layout: StreamLayout,
                device: MTLDevice,
                slotCount: Int,
                cachePolicy: ExpertCachePolicy = .lfu,
                eventCoordinator: ExpertIOEventCoordinator? = nil,
                metalStagingPool: MetalExpertStagingPool? = nil,
                metalIOService: MetalExpertIOService? = nil) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        if let rawPolicy = ProcessInfo.processInfo.environment["NVMAI_EXPERT_CACHE_POLICY"] {
            guard let experimentalPolicy = ExpertCachePolicy(rawValue: rawPolicy) else {
                throw ModelError.internalInconsistency(
                    detail: "unsupported NVMAI_EXPERT_CACHE_POLICY '\(rawPolicy)'; allowed: lfu, lru, aging-lfu")
            }
            self.cachePolicy = experimentalPolicy
        } else {
            self.cachePolicy = cachePolicy
        }
        self.eventCoordinator = eventCoordinator
        self.metalStagingPool = metalStagingPool
        self.metalIOService = metalIOService
        self.ioBackend = try ExpertIOBackend.environmentValue()
        self.cacheLayout = try ExpertCacheLayout.environmentValue()
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
        // The pool base retains the validated 2 MiB allocation alignment.
        // Individual offsets need only VM-page alignment for pread and Metal;
        // rounding every slot to 2 MiB inflated the 8-bit pool by several GiB.
        self.poolSlotStride = allocationSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        var bufferOffsets: [UInt64] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)
        bufferOffsets.reserveCapacity(slotCount)
        guard let residencyTable = device.makeBuffer(
            length: max(1, layout.expertsPerLayer)
                * MemoryLayout<ExpertResidencyEntry>.stride,
            options: .storageModeShared)
        else {
            close(openedFD)
            throw StreamerError.bufferWrapFailed
        }
        self.residencyTable = residencyTable
        let residencyEntries = residencyTable.contents()
            .bindMemory(to: ExpertResidencyEntry.self,
                        capacity: max(1, layout.expertsPerLayer))
        for expert in 0..<max(1, layout.expertsPerLayer) {
            residencyEntries[expert] = ExpertResidencyEntry()
        }

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
            close(openedFD)
        }

        if cacheLayout == .pool {
            let (poolBytes, overflow) = poolSlotStride.multipliedReportingOverflow(by: slotCount)
            guard !overflow else {
                unwind()
                throw StreamerError.allocFailed(errno: EOVERFLOW)
            }
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, poolBytes)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: poolBytes,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            for slot in 0..<slotCount {
                if slot > 0 {
                    pointers.append(pointer.advanced(by: slot * poolSlotStride))
                }
                buffers.append(buffer)
                bufferOffsets.append(UInt64(slot * poolSlotStride))
            }
        } else {
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
                bufferOffsets.append(0)
            }
        }

        // Fail closed when bounded I/O was requested. Falling through to an
        // ordinary descriptor would silently create an unbounded second cache
        // in the macOS page cache and invalidate the declared RAM budget.
        if ioBackend == .metal {
            do {
                if let metalIOService {
                    self.metalReader = try MetalExpertReader(
                        path: layout.path, device: device, service: metalIOService)
                } else {
                    // Direct construction remains useful for focused tests;
                    // Model opens pass the one shared service above.
                    self.metalReader = try MetalExpertReader(
                        path: layout.path, device: device, maximumCommandsInFlight: 4)
                }
            } catch {
                unwind()
                throw error
            }
            self.boundedReader = nil
        } else if ProcessInfo.processInfo.environment["NVMAI_BOUNDED_IO"] != "0" {
            self.metalReader = nil
            do {
                self.boundedReader = try ParallelExpertReader(
                    path: layout.path,
                    expertStride: Int(layout.expertStride),
                    threads: 4,
                    bypassCache: true)
            } catch {
                unwind()
                throw error
            }
        } else {
            self.metalReader = nil
            self.boundedReader = nil
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotBufferOffsets = bufferOffsets
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.slotState = [SlotState](repeating: .empty, count: slotCount)
        self.slotGeneration = [UInt64](repeating: 0, count: slotCount)
        self.slotPinCount = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.expertLoadCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        // K12: slot selection and fill share one critical section so the
        // round-robin path never lands on a slot a concurrent plan reserved
        // (`loading`) and no fill can interleave with another pread.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        var candidate = nextSlot
        var scanned = 0
        while (slotState[candidate] == .loading || slotPinCount[candidate] > 0)
            && scanned < slotCount {
            candidate = (candidate + 1) % slotCount
            scanned += 1
        }
        guard scanned < slotCount else {
            throw ModelError.expertCacheUnplaceable(
                detail: "all \(slotCount) expert-cache slots are loading or pinned")
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
        guard slotState[slot] != .loading, slotPinCount[slot] == 0 else {
            throw ModelError.expertCacheUnplaceable(
                detail: "expert-cache slot \(slot) is loading or pinned")
        }
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
        slotGeneration[slot] &+= 1
        let previousExpert = slotExpert[slot]
        if previousExpert >= 0 {
            publishResidencyUnlocked(expert: previousExpert,
                                     slot: slot,
                                     state: ExpertResidencyEntry.empty,
                                     generation: slotGeneration[slot])
        }
        slotExpert[slot] = expert
        slotState[slot] = .loading
        publishResidencyUnlocked(expert: expert,
                                 slot: slot,
                                 state: ExpertResidencyEntry.loading,
                                 generation: slotGeneration[slot])
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            try readFull(
                into: slotPointers[slot],
                fileOffset: layout.streamOffset + regionOffset,
                count: Int(layout.expertStride))
            slotState[slot] = .resident
            publishResidencyUnlocked(expert: expert,
                                     slot: slot,
                                     state: ExpertResidencyEntry.resident,
                                     generation: slotGeneration[slot])
            slotLastUse[slot] = useClock
            recordSuccessfulLoadsUnlocked(
                experts: [expert], elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
        } catch {
            slotState[slot] = .empty
            slotExpert[slot] = -1
            publishResidencyUnlocked(expert: expert,
                                     slot: slot,
                                     state: ExpertResidencyEntry.empty,
                                     generation: slotGeneration[slot])
            throw error
        }
        return (slotBuffers[slot], slotBufferOffsets[slot], layout.expertStride)
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
        if cachePolicy == .agingLFU,
           statisticsPlans > 0,
           statisticsPlans.isMultiple(of: 1_024) {
            for expert in expertUseCount.indices {
                expertUseCount[expert] >>= 1
            }
        }
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)
        // Loading slots are not valid hits and cannot be reassigned.
        for slot in 0..<slotCount where slotState[slot] == .loading {
            reserved[slot] = true
        }

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotState[slot] == .resident
                    && slotExpert[slot] == experts[index] {
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
            .filter { !reserved[$0] && slotState[$0] != .loading && slotPinCount[$0] == 0 }
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
            if slotState[slot] == .resident { statisticsEvictions &+= 1 }
            let previousExpert = slotExpert[slot]
            assignedSlots[index] = slot
            reserved[slot] = true
            slotGeneration[slot] &+= 1
            slotExpert[slot] = experts[index]
            slotLastUse[slot] = clock
            slotState[slot] = .loading
            if previousExpert >= 0 {
                publishResidencyUnlocked(expert: previousExpert,
                                         slot: slot,
                                         state: ExpertResidencyEntry.empty,
                                         generation: slotGeneration[slot])
            }
            publishResidencyUnlocked(expert: experts[index],
                                     slot: slot,
                                     state: ExpertResidencyEntry.loading,
                                     generation: slotGeneration[slot])
        }

        statisticsPlans &+= 1
        statisticsRequestedExperts &+= UInt64(experts.count)
        statisticsHits &+= UInt64(experts.count - misses.count)
        statisticsMisses &+= UInt64(misses.count)
        statisticsPeakLoadingSlots = max(
            statisticsPeakLoadingSlots,
            slotState.count(where: { $0 == .loading }))

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            assignedGenerations: assignedSlots.map { slotGeneration[$0] },
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
        precondition(plan.assignedGenerations.count == plan.experts.count,
                     "expert cache plan generation count mismatch")

        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var succeeded = false
        defer {
            finishPlanExecution(
                plan,
                succeeded: succeeded,
                elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
        }

        if !plan.misses.isEmpty {
            if let metalReader {
                try executeMetalReads(plan, reader: metalReader)
            } else if let boundedReader {
                try executeBoundedReads(plan, reader: boundedReader)
            } else {
                let parallel = ProcessInfo.processInfo.environment["NVMAI_PARALLEL_IO"] != "0"
                    && plan.misses.count > 1
                try executeCachedPreads(plan, parallel: parallel)
            }
            try markPlanMissesResident(plan)
        }

        succeeded = true
        return expertCachePlanBuffers(plan)
    }

    /// Submits the plan to the persistent storage service and returns before
    /// any read has to complete. Reserved generations are already pinned by
    /// the caller, so the destination pointers remain valid for the operation.
    public func beginExpertCachePlan(
        _ plan: ExpertCachePlan,
        eventDriven: Bool = false
    ) throws -> ExpertLoadOperation {
        let token: ExpertIOCompletionToken?
        if eventDriven {
            guard let eventCoordinator else {
                throw ModelError.internalInconsistency(
                    detail: "event-driven expert I/O requested without a shared event")
            }
            token = try eventCoordinator.reserve()
        } else {
            token = nil
        }
        guard !plan.misses.isEmpty else {
            let operation = ExpertLoadOperation(
                completionToken: token,
                eventCoordinator: eventCoordinator,
                backendSignalsEvent: false)
            operation.finish(.success(()))
            return operation
        }
        if let metalReader {
            if eventDriven {
                return try beginEventDrivenMetalReads(
                    plan, reader: metalReader, token: token)
            }
            let operation = ExpertLoadOperation(
                completionToken: token,
                eventCoordinator: eventCoordinator,
                backendSignalsEvent: false)
            operation.markInFlight()
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            do {
                try beginMetalReads(
                    plan,
                    reader: metalReader,
                    // A native MTLIO signal cross-queued with a waiting compute
                    // buffer deadlocked on the qualification M3. Keep Metal I/O
                    // nonblocking, but bridge its completion handler through
                    // the same proven coordinator used by bounded pread.
                    completionToken: nil) { [self, operation] result in
                        switch result {
                        case .success:
                            do {
                                try markPlanMissesResident(plan)
                                finishPlanExecution(
                                    plan,
                                    succeeded: true,
                                    elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                                operation.finish(.success(()))
                            } catch {
                                finishPlanExecution(
                                    plan,
                                    succeeded: false,
                                    elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                                operation.finish(.failure(error))
                            }
                        case .failure(let error):
                            finishPlanExecution(
                                plan,
                                succeeded: false,
                                elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                            operation.finish(.failure(error))
                        }
                    }
            } catch {
                finishPlanExecution(plan, succeeded: false, elapsedNanos: 0)
                operation.finish(.failure(error))
            }
            return operation
        }
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: eventCoordinator,
            backendSignalsEvent: false)
        ExpertIOScheduler.shared.submit { [self, operation] in
            operation.markInFlight()
            do {
                _ = try executeExpertCachePlan(plan)
                operation.finish(.success(()))
            } catch {
                operation.finish(.failure(error))
            }
        }
        return operation
    }

    private func beginEventDrivenMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader,
        token: ExpertIOCompletionToken?
    ) throws -> ExpertLoadOperation {
        guard let token,
              let stagingLease = metalStagingPool?.tryAcquire(count: plan.misses.count)
        else {
            throw ModelError.internalInconsistency(
                detail: "event-driven Metal I/O staging ring is unavailable")
        }
        let transfer = try makeMetalStagingTransfer(plan: plan, stagingLease: stagingLease)
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: eventCoordinator,
            // MTLIO writes the status word and signals the event in command
            // order. Its handler records terminal state but never wakes the
            // decode task to encode a fixup.
            backendSignalsEvent: true,
            metalStagingTransfer: transfer,
            requiresGPUFinalization: true)
        operation.markInFlight()
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            try beginMetalReads(
                plan,
                reader: reader,
                destinations: stagingLease.buffers,
                destinationOffsets: [Int](repeating: 0, count: stagingLease.buffers.count),
                completionToken: token) { [self, operation] result in
                    switch result {
                    case .success:
                        // Cache slots remain LOADING. The runner publishes
                        // RESIDENT only after its event-gated blit completes.
                        finishPlanExecution(
                            plan,
                            succeeded: true,
                            elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                        operation.finish(.success(()))
                    case .failure(let error):
                        finishPlanExecution(
                            plan,
                            succeeded: false,
                            elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                        operation.finish(.failure(error))
                    }
                }
        } catch {
            finishPlanExecution(plan, succeeded: false, elapsedNanos: 0)
            operation.releaseStagingTransfer()
            operation.finish(.failure(error))
        }
        return operation
    }

    private func executeMetalReads(_ plan: ExpertCachePlan,
                                   reader: MetalExpertReader) throws {
        var offsets: [UInt64] = []
        var destinations: [MTLBuffer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotBuffers[plan.assignedSlots[index]])
        }
        try reader.fetch(
            offsets: offsets,
            into: destinations,
            byteCount: Int(layout.expertStride),
            destinationOffsets: plan.misses.map {
                Int(slotBufferOffsets[plan.assignedSlots[$0]])
            })
    }

    private func beginMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader,
        destinations explicitDestinations: [MTLBuffer]? = nil,
        destinationOffsets explicitDestinationOffsets: [Int]? = nil,
        completionToken: ExpertIOCompletionToken?,
        completion: @escaping @Sendable (Result<Void, any Error>) -> Void
    ) throws {
        var offsets: [UInt64] = []
        var destinations: [MTLBuffer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotBuffers[plan.assignedSlots[index]])
        }
        let finalDestinations = explicitDestinations ?? destinations
        let finalDestinationOffsets = explicitDestinationOffsets ?? plan.misses.map {
            Int(slotBufferOffsets[plan.assignedSlots[$0]])
        }
        try reader.beginFetch(
            offsets: offsets,
            into: finalDestinations,
            byteCount: Int(layout.expertStride),
            destinationOffsets: finalDestinationOffsets,
            completionToken: completionToken,
            completion: completion)
    }

    private func makeMetalStagingTransfer(
        plan: ExpertCachePlan,
        stagingLease: MetalExpertStagingLease
    ) throws -> MetalExpertStagingTransfer {
        var destinations: [MTLBuffer] = []
        var destinationOffsets: [Int] = []
        destinations.reserveCapacity(plan.misses.count)
        destinationOffsets.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotBuffers.count else {
                stagingLease.release()
                throw ModelError.internalInconsistency(
                    detail: "Metal I/O staging transfer references an invalid cache slot")
            }
            destinations.append(slotBuffers[slot])
            destinationOffsets.append(Int(slotBufferOffsets[slot]))
        }
        return MetalExpertStagingTransfer(
            lease: stagingLease,
            destinations: destinations,
            destinationOffsets: destinationOffsets,
            byteCount: Int(layout.expertStride))
    }

    private func executeBoundedReads(_ plan: ExpertCachePlan,
                                     reader: ParallelExpertReader) throws {
        var offsets: [UInt64] = []
        var destinations: [UnsafeMutableRawPointer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotPointers[plan.assignedSlots[index]])
        }
        try reader.fetch(offsets: offsets, into: destinations)
    }

    private func executeCachedPreads(_ plan: ExpertCachePlan,
                                     parallel: Bool) throws {
        if parallel {
            let firstError = Mutex<Error?>(nil)
            DispatchQueue.concurrentPerform(iterations: plan.misses.count) { offset in
                do {
                    try readPlanMiss(plan, index: plan.misses[offset])
                } catch {
                    firstError.withLock { if $0 == nil { $0 = error } }
                }
            }
            if let error = firstError.withLock({ $0 }) { throw error }
            return
        }
        for index in plan.misses { try readPlanMiss(plan, index: index) }
    }

    private func readPlanMiss(_ plan: ExpertCachePlan, index: Int) throws {
        try readFull(
            into: slotPointers[plan.assignedSlots[index]],
            fileOffset: try fileOffset(plan: plan, index: index),
            count: Int(layout.expertStride))
    }

    private func fileOffset(plan: ExpertCachePlan, index: Int) throws -> UInt64 {
        let regionOffset = layout.expertOffset(
            layer: plan.layer,
            expert: plan.experts[index])
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        return layout.streamOffset + regionOffset
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], slotBufferOffsets[slot], layout.expertStride)
        }
    }

    public func expertResidencyResources() -> ExpertResidencyResources {
        ExpertResidencyResources(
            table: residencyTable,
            expertPool: cacheLayout == .pool ? slotBuffers.first : nil,
            poolSlotStride: UInt64(poolSlotStride),
            expertStride: layout.expertStride,
            expertCount: layout.expertsPerLayer)
    }

    public func residencyEntry(expert: Int) -> ExpertResidencyEntry {
        precondition(expert >= 0 && expert < layout.expertsPerLayer)
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return residencyTable.contents()
            .bindMemory(to: ExpertResidencyEntry.self,
                        capacity: layout.expertsPerLayer)[expert]
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
        let misses = experts.filter { expert in
            !slotExpert.indices.contains { slot in
                slotState[slot] == .resident && slotExpert[slot] == expert
            }
        }
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

    func pin(_ plan: ExpertCachePlan) throws -> ExpertCacheLease {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard plan.assignedSlots.count == plan.experts.count,
              plan.assignedGenerations.count == plan.experts.count else {
            throw ModelError.internalInconsistency(
                detail: "cannot pin an incomplete expert-cache plan")
        }
        for index in plan.experts.indices {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotCount,
                  slotGeneration[slot] == plan.assignedGenerations[index],
                  slotExpert[slot] == plan.experts[index],
                  slotState[slot] != .empty else {
                throw ModelError.internalInconsistency(
                    detail: "expert-cache plan became stale before GPU pin")
            }
        }
        for slot in plan.assignedSlots { slotPinCount[slot] &+= 1 }
        return ExpertCacheLease(
            streamer: self,
            slots: plan.assignedSlots,
            generations: plan.assignedGenerations)
    }

    fileprivate func unpin(slots: [Int], generations: [UInt64]) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for (slot, generation) in zip(slots, generations)
            where slot >= 0 && slot < slotCount && slotGeneration[slot] == generation {
            precondition(slotPinCount[slot] > 0, "expert-cache slot pin underflow")
            slotPinCount[slot] -= 1
        }
    }

    public func statistics() -> ExpertStreamingStatistics {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return ExpertStreamingStatistics(
            plans: statisticsPlans,
            requestedExperts: statisticsRequestedExperts,
            hits: statisticsHits,
            misses: statisticsMisses,
            bytesRead: statisticsBytesRead,
            readOperations: statisticsReadOperations,
            evictions: statisticsEvictions,
            reloads: statisticsReloads,
            loadBatches: statisticsLoadBatches,
            totalLoadNanos: statisticsTotalLoadNanos,
            maximumLoadNanos: statisticsMaximumLoadNanos,
            latencyHistogram: statisticsLatencyHistogram,
            residentSlots: slotState.count(where: { $0 == .resident }),
            loadingSlots: slotState.count(where: { $0 == .loading }),
            pinnedSlots: slotPinCount.count(where: { $0 > 0 }),
            peakLoadingSlots: statisticsPeakLoadingSlots)
    }

    private func finishPlanExecution(_ plan: ExpertCachePlan,
                                     succeeded: Bool,
                                     elapsedNanos: UInt64) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if succeeded {
            recordSuccessfulLoadsUnlocked(
                experts: plan.misses.map { plan.experts[$0] },
                elapsedNanos: elapsedNanos)
            return
        }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            if slotGeneration[slot] == plan.assignedGenerations[index],
               slotState[slot] == .loading {
                slotState[slot] = .empty
                slotExpert[slot] = -1
                publishResidencyUnlocked(expert: plan.experts[index],
                                         slot: slot,
                                         state: ExpertResidencyEntry.empty,
                                         generation: plan.assignedGenerations[index])
            }
        }
    }

    private func markPlanMissesResident(_ plan: ExpertCachePlan) throws {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slotGeneration[slot] == plan.assignedGenerations[index] else {
                throw ModelError.internalInconsistency(
                    detail: "expert-cache slot generation changed during expert load")
            }
        }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            slotState[slot] = .resident
            slotExpert[slot] = plan.experts[index]
            slotLastUse[slot] = useClock
            publishResidencyUnlocked(expert: plan.experts[index],
                                     slot: slot,
                                     state: ExpertResidencyEntry.resident,
                                     generation: plan.assignedGenerations[index])
        }
    }

    /// The Metal staging route copies into cache slots on the GPU after the
    /// MTLIO event. Its slots cannot become resident until that command buffer
    /// has completed, otherwise a later layer could read bytes still owned by
    /// the blit engine.
    func markStagedMetalPlanResident(_ plan: ExpertCachePlan) throws {
        try markPlanMissesResident(plan)
    }

    /// Clears a staged load if its event-gated transfer command fails. This is
    /// intentionally separate from `finishPlanExecution`: I/O may have
    /// succeeded and been accounted for, while the GPU copy did not complete.
    func failStagedMetalPlan(_ plan: ExpertCachePlan) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotCount,
                  slotGeneration[slot] == plan.assignedGenerations[index],
                  slotState[slot] == .loading else { continue }
            slotState[slot] = .empty
            slotExpert[slot] = -1
            publishResidencyUnlocked(expert: plan.experts[index],
                                     slot: slot,
                                     state: ExpertResidencyEntry.empty,
                                     generation: plan.assignedGenerations[index])
        }
    }

    /// CPU publication occurs under the cache lock. A loading entry is visible
    /// immediately after reservation; resident is written only after every
    /// byte lands. Event-driven consumers additionally wait on the batch's
    /// shared-event value, which is the CPU/GPU release/acquire boundary.
    private func publishResidencyUnlocked(expert: Int,
                                          slot: Int,
                                          state: UInt32,
                                          generation: UInt64) {
        guard expert >= 0 && expert < layout.expertsPerLayer else { return }
        let entries = residencyTable.contents()
            .bindMemory(to: ExpertResidencyEntry.self,
                        capacity: layout.expertsPerLayer)
        entries[expert] = ExpertResidencyEntry(
            slot: state == ExpertResidencyEntry.empty
                ? ExpertResidencyEntry.notResidentSlot : UInt32(slot),
            state: state,
            generation: generation)
    }

    private func recordSuccessfulLoadsUnlocked(experts: [Int], elapsedNanos: UInt64) {
        guard !experts.isEmpty else { return }
        statisticsBytesRead &+= UInt64(experts.count) * layout.expertStride
        statisticsReadOperations &+= UInt64(experts.count)
        statisticsLoadBatches &+= 1
        statisticsTotalLoadNanos &+= elapsedNanos
        statisticsMaximumLoadNanos = max(statisticsMaximumLoadNanos, elapsedNanos)
        let bucket = Self.latencyBucketIndex(nanos: elapsedNanos)
        statisticsLatencyHistogram[bucket] &+= 1
        for expert in experts where expert >= 0 && expert < expertLoadCount.count {
            if expertLoadCount[expert] > 0 { statisticsReloads &+= 1 }
            expertLoadCount[expert] &+= 1
        }
    }

    private static func latencyBucketIndex(nanos: UInt64) -> Int {
        var bound: UInt64 = 125_000
        for index in 0..<16 {
            if nanos <= bound { return index }
            bound &*= 2
        }
        return 16
    }

    static func latencyBucketUpperBound(index: Int) -> UInt64 {
        guard index < 16 else { return UInt64.max }
        return 125_000 << UInt64(index)
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

/// Pins exact slot generations until every GPU command using them completes.
/// Release is idempotent so error cleanup and normal command completion can
/// safely converge on the same lifetime operation.
/// unchecked-invariant: immutable slot metadata is published at init and the
/// only mutable release flag is guarded by `lock`; streamer state has its own lock.
final class ExpertCacheLease: @unchecked Sendable {
    private weak var streamer: PreadExpertStreamer?
    private let slots: [Int]
    private let generations: [UInt64]
    private let lock = NSLock()
    private var released = false

    fileprivate init(streamer: PreadExpertStreamer,
                     slots: [Int],
                     generations: [UInt64]) {
        self.streamer = streamer
        self.slots = slots
        self.generations = generations
    }

    func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        lock.unlock()
        streamer?.unpin(slots: slots, generations: generations)
    }

    deinit { release() }
}
