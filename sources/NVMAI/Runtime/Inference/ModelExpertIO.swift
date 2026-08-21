import Foundation
import Metal

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

/// unchecked-invariant: the wrapped cache lease is thread-safe and immutable;
/// this forwarding owner adds no mutable state.
final class RoutedExpertLease: @unchecked Sendable {
    private let cacheLease: ExpertCacheLease

    init(cacheLease: ExpertCacheLease) {
        self.cacheLease = cacheLease
    }

    func release() { cacheLease.release() }
}

/// Model-level storage ticket. Expert views are derivable from the reserved
/// plan immediately; callers must await `completion()` before consuming miss
/// slots unless a GPU event dependency enforces the same ordering.
/// unchecked-invariant: immutable model, plan, and thread-safe storage ticket.
public final class RoutedExpertLoadOperation: @unchecked Sendable {
    public let plan: RoutedExpertFetchPlan
    public let storage: ExpertLoadOperation
    private let model: Model

    init(model: Model,
         plan: RoutedExpertFetchPlan,
         storage: ExpertLoadOperation) {
        self.model = model
        self.plan = plan
        self.storage = storage
    }

    public var state: ExpertLoadOperationState { storage.state }

    public func wait() throws -> [TensorView] {
        try storage.wait()
        return try model.routedExpertBuffers(for: plan)
    }

    public func completion() async throws -> [TensorView] {
        try await storage.completion()
        return try model.routedExpertBuffers(for: plan)
    }
}

extension Model {
    public func routedExpertStatistics() -> ExpertStreamingStatistics {
        let streamers = streamersQueue.sync { streamersBox.streamers.compactMap { $0 } }
        return streamers.reduce(.zero) { $0.adding($1.statistics()) }
    }

    public func routedExpertOffsets(layer: Int) throws -> MoEExpertOffsets {
        let expert = try packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            UInt32(expert.subTensors[role]?.offset ?? 0)
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: try streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots))
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    /// Cache slot count is a per-model streaming property (the same for every
    /// layer), so it deliberately takes no layer argument.
    public func routedExpertCacheSlotCount() -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func routedExpertResidency(layer: Int) throws -> ExpertResidencyResources {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.expertResidencyResources()
    }

    func pinRoutedExperts(for plan: RoutedExpertFetchPlan) throws -> RoutedExpertLease {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return RoutedExpertLease(cacheLease: try streamer.pin(plan.cachePlan))
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try await beginFetchRoutedExperts(plan: plan).completion()
    }

    public func beginFetchRoutedExperts(
        plan: RoutedExpertFetchPlan,
        eventDriven: Bool = false
    ) throws -> RoutedExpertLoadOperation {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return RoutedExpertLoadOperation(
            model: self,
            plan: plan,
            storage: try streamer.beginExpertCachePlan(
                plan.cachePlan,
                eventDriven: eventDriven))
    }

    /// Publishes the cache slots filled by an event-gated Metal staging copy.
    /// Call only after the command buffer that copied staging into the slots
    /// has completed; until then the cache deliberately reports these experts
    /// as `LOADING` to both CPU and GPU residency lookups.
    func finalizeRoutedExpertStagingTransfer(
        plan: RoutedExpertFetchPlan
    ) throws {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        try streamer.markStagedMetalPlanResident(plan.cachePlan)
    }

    /// Clears a staged plan whose dependent GPU command failed before its
    /// staging bytes could become a valid cache entry.
    func failRoutedExpertStagingTransfer(
        plan: RoutedExpertFetchPlan
    ) {
        guard (try? ensureLayerOpened(plan.layer)) != nil else { return }
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        streamer.failStagedMetalPlan(plan.cachePlan)
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.loadExpertsCached(experts: experts)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: 0)
        }
    }
}
