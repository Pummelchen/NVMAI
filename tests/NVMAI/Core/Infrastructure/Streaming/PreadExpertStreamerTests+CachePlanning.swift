import Darwin
import Foundation
import Metal
import Testing

@testable import NVMAI

extension PreadExpertStreamerTests {
  @Test func expertIOBackendEnvironmentDefaultsAndFailsClosed() throws {
    #expect(try ExpertIOBackend.environmentValue([:]) == .pread)
    #expect(try ExpertIOBackend.environmentValue([
      "NVMAI_EXPERT_IO_BACKEND": "metal",
    ]) == .metal)
    #expect(throws: (any Error).self) {
      try ExpertIOBackend.environmentValue([
        "NVMAI_EXPERT_IO_BACKEND": "unknown",
      ])
    }
  }

  @Test func cachedBatchWithoutExecutorLoadsTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let results = try streamer.loadExpertsCached(experts: [3, 1, 2])
    for (index, result) in results.enumerated() {
      let expert = [3, 1, 2][index]
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  @Test func adviseExpertsDoesNotChangeLoadedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    let experts = [0, 2, 3]

    let advice = streamer.adviseExperts(experts: experts)
    #expect(advice.requested == experts.count)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == experts.count)
    #endif

    let results = try streamer.loadExpertsCached(experts: experts)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func adviseExpertMissesSkipsResidentSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let advice = streamer.adviseExpertMisses(experts: [0, 1, 2])

    #expect(advice.requested == 2)
    #expect(advice.calls == 1)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == 1)
    #endif
  }

  @Test func plannedCacheLoadExecutesSameMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = try streamer.planExpertsCached(experts: experts)

    #expect(plan.hits == 1)
    #expect(plan.misses.map { experts[$0] } == [1, 2])

    let results = try streamer.executeExpertCachePlan(plan)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func residentSnapshotExcludesLoadingEntries() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [1, 3])
    #expect(streamer.residentExperts() == [1, 3])

    let plan = try streamer.planExpertsCached(experts: [2])
    #expect(plan.misses == [0])
    #expect(streamer.residentExperts() == [1, 3])

    _ = try streamer.executeExpertCachePlan(plan)
    #expect(streamer.residentExperts() == [1, 2, 3])
  }

  @Test func submittedCacheLoadCompletesWithoutCallerExecutingReads() async throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let context = try MetalContext()
    let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: context.device,
      slotCount: 4, eventCoordinator: coordinator)
    let plan = try streamer.planExpertsCached(experts: [3, 1, 2])

    let operation = try streamer.beginExpertCachePlan(plan, eventDriven: true)
    try await operation.completion()

    #expect(operation.state == .completed)
    let token = try #require(operation.completionToken)
    #expect(token.event.signaledValue >= token.value)
    for (index, result) in streamer.expertCachePlanBuffers(plan).enumerated() {
      let got = Self.bytes(of: result.buffer, offset: result.offset,
                           count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(plan.experts[index]) })
    }
  }

  @Test func contiguousPoolUsesAlignedNonOverlappingSlotOffsets() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    setenv("NVMAI_EXPERT_CACHE_LAYOUT", "pool", 1)
    defer { unsetenv("NVMAI_EXPERT_CACHE_LAYOUT") }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    let plan = try streamer.planExpertsCached(experts: [0, 1, 2, 3])
    let operation = try streamer.beginExpertCachePlan(plan)
    try operation.wait()
    let buffers = streamer.expertCachePlanBuffers(plan)

    #expect(streamer.cacheLayout == .pool)
    #expect(buffers.allSatisfy { $0.buffer === buffers[0].buffer })
    #expect(Set(buffers.map(\.offset)).count == 4)
    #expect(buffers.allSatisfy { Int($0.offset).isMultiple(of: Int(getpagesize())) })
    for (index, result) in buffers.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: result.offset,
                           count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(index) })
    }
  }

  @Test func plannedCacheBuffersExposeReservedSlotsBeforeExecute() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = try streamer.planExpertsCached(experts: experts)
    let reserved = streamer.expertCachePlanBuffers(plan)

    let hitBytes = Self.bytes(of: reserved[0].buffer, offset: 0, count: Self.expertStride)
    #expect(hitBytes.allSatisfy { $0 == Self.tagByte(0) })

    let executed = try streamer.executeExpertCachePlan(plan)
    for i in 0..<experts.count {
      #expect(reserved[i].buffer === executed[i].buffer)
      let got = Self.bytes(of: executed[i].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[i]) })
    }
  }

  @Test func plannedCacheAvoidsInFlightSlotsForHitsAndMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let warmed = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = try streamer.planExpertsCached(
      experts: [0, 2],
      avoidingSlots: [0, 1])

    #expect(plan.assignedSlots == [0, 2])
    #expect(plan.hits == 1)
    #expect(plan.misses == [1])

    let executed = try streamer.executeExpertCachePlan(plan)
    for (index, expert) in plan.experts.enumerated() {
      let got = Self.bytes(of: executed[index].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }

    let avoidedBytes = Self.bytes(of: warmed[0].buffer, offset: 0, count: Self.expertStride)
    #expect(avoidedBytes.allSatisfy { $0 == Self.tagByte(0) })
  }

  @Test func plannedCacheReturnsNilWhenMissesCannotAvoidInFlightSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCachedIfPossible(
      experts: [0, 2, 3, 4],
      avoidingSlots: [0, 1])

    #expect(plan == nil)
  }

  @Test func pinnedGenerationsCannotBeEvictedUntilReleased() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let residentPlan = try streamer.planExpertsCached(experts: [0, 1])
    let lease = try streamer.pin(residentPlan)

    #expect(streamer.statistics().pinnedSlots == 2)
    #expect(streamer.planExpertsCachedIfPossible(experts: [2]) == nil)

    lease.release()
    #expect(streamer.statistics().pinnedSlots == 0)
    let replacement = try streamer.planExpertsCached(experts: [2])
    _ = try streamer.executeExpertCachePlan(replacement)
    #expect(streamer.statistics().residentSlots == 2)
  }

  @Test func streamingStatisticsCountHitsMissesReloadsAndBoundLatency() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    _ = try streamer.loadExpertsCached(experts: [0, 2])
    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let statistics = streamer.statistics()

    #expect(statistics.plans == 3)
    #expect(statistics.requestedExperts == 6)
    #expect(statistics.hits == 2)
    #expect(statistics.misses == 4)
    #expect(statistics.readOperations == 4)
    #expect(statistics.bytesRead == 4 * UInt64(Self.expertStride))
    #expect(statistics.evictions == 2)
    #expect(statistics.reloads == 1)
    #expect(statistics.loadBatches == 3)
    #expect(statistics.latencyHistogram.reduce(0, +) == 3)
    #expect(statistics.loadLatencyPercentile(0.5) > 0)
    #expect(statistics.loadingSlots == 0)
  }

  @Test func failedPlannedReadReleasesLoadingSlot() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)
    let plan = try streamer.planExpertsCached(experts: [3])

    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: Self.streamOffset + UInt64(Self.expertStride))
    try handle.close()
    #expect(throws: (any Error).self) {
      _ = try streamer.executeExpertCachePlan(plan)
    }
    #expect(streamer.statistics().loadingSlots == 0)
  }

  @Test func concurrentPinnedPlansNeverExposeOverwrittenBytes() async throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    try await withThrowingTaskGroup(of: Void.self) { group in
      for worker in 0..<2 {
        group.addTask {
          for iteration in 0..<50 {
            let expert = (worker * 2 + iteration) % Self.numExperts
            let plan = try streamer.planExpertsCached(experts: [expert])
            let lease = try streamer.pin(plan)
            defer { lease.release() }
            let result = try streamer.executeExpertCachePlan(plan)[0]
            let bytes = Self.bytes(
              of: result.buffer, offset: result.offset, count: Self.expertStride)
            guard bytes.allSatisfy({ $0 == Self.tagByte(expert) }) else {
              throw ModelError.internalInconsistency(
                detail: "pinned expert slot was overwritten")
            }
          }
        }
      }
      try await group.waitForAll()
    }
    #expect(streamer.statistics().pinnedSlots == 0)
    #expect(streamer.statistics().loadingSlots == 0)
  }

}
