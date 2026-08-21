import Foundation
import Metal

/// A fixed raw-byte staging ring for v4.3 predictive routed-expert reads.
///
/// Slots are intentionally outside the authoritative expert cache. A completed
/// entry becomes a cache resident only when the exact router selects it and the
/// ordinary cache planner adopts its bytes. Failed or incorrect predictions
/// are simply discarded without changing cache mappings or slot generations.
/// unchecked-invariant: slot ownership and operation association are guarded
/// by `lock`, and backing buffers are retained while an operation can write.
final class ExpertPrefetchRing: @unchecked Sendable {
    private struct Slot {
        let buffer: MTLBuffer
        var layer = -1
        var expert = -1
        var operation: ExpertLoadOperation?
    }

    private let lock = NSLock()
    private var slots: [Slot]

    init(device: MTLDevice, expertStride: Int, slotCount: Int) throws {
        guard expertStride > 0, slotCount > 0 else {
            throw ModelError.internalInconsistency(detail: "invalid prefetch ring geometry")
        }
        var allocated: [Slot] = []
        allocated.reserveCapacity(slotCount)
        for index in 0..<slotCount {
            guard let buffer = device.makeBuffer(length: expertStride,
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "decode.prefetch.\(index)"
            allocated.append(Slot(buffer: buffer))
        }
        slots = allocated
    }

    /// Begins missing next-layer reads in unused staging slots. Already queued
    /// predictions are deduplicated; finished entries are reclaimed only when
    /// they have not been selected by a later exact route.
    func begin(model: Model, layer: Int, experts: [Int], resident: Set<Int>) throws {
        lock.lock()
        reclaimTerminalSlotsUnlocked(exceptLayer: layer)
        let active = Set(slots.compactMap { slot in
            slot.layer == layer && slot.expert >= 0 ? slot.expert : nil
        })
        var seen: Set<Int> = []
        let wanted = experts.filter {
            !resident.contains($0) && !active.contains($0) && seen.insert($0).inserted
        }
        let free = slots.indices.filter { slots[$0].expert < 0 }
        let count = min(wanted.count, free.count)
        guard count > 0 else {
            lock.unlock()
            return
        }
        let selectedSlots = Array(free.prefix(count))
        let selectedExperts = Array(wanted.prefix(count))
        let buffers = selectedSlots.map { slots[$0].buffer }
        for (slot, expert) in zip(selectedSlots, selectedExperts) {
            slots[slot].layer = layer
            slots[slot].expert = expert
            slots[slot].operation = nil
        }
        lock.unlock()

        do {
            let operation = try model.beginRoutedExpertPrefetch(
                layer: layer, experts: selectedExperts, into: buffers)
            lock.lock()
            for slot in selectedSlots { slots[slot].operation = operation }
            lock.unlock()
        } catch {
            lock.lock()
            for slot in selectedSlots {
                slots[slot].layer = -1
                slots[slot].expert = -1
                slots[slot].operation = nil
            }
            lock.unlock()
            throw error
        }
    }

    /// Completed raw bytes for one exact route. In-flight reads are not
    /// awaited: a demand miss remains authoritative and may start immediately.
    func readyBuffers(layer: Int, experts: [Int]) -> [Int: MTLBuffer] {
        let requested = Set(experts)
        lock.lock()
        defer { lock.unlock() }
        var result: [Int: MTLBuffer] = [:]
        for slot in slots where slot.layer == layer && requested.contains(slot.expert) {
            if slot.operation?.state == .completed {
                result[slot.expert] = slot.buffer
            }
        }
        return result
    }

    func consume(layer: Int, experts: Set<Int>) {
        lock.lock()
        defer { lock.unlock() }
        for index in slots.indices where slots[index].layer == layer
            && experts.contains(slots[index].expert) {
            slots[index].layer = -1
            slots[index].expert = -1
            slots[index].operation = nil
        }
    }

    private func reclaimTerminalSlotsUnlocked(exceptLayer: Int) {
        for index in slots.indices where slots[index].layer != exceptLayer {
            switch slots[index].operation?.state {
            case .completed, .failed, .none:
                slots[index].layer = -1
                slots[index].expert = -1
                slots[index].operation = nil
            case .submitted, .inFlight:
                break
            }
        }
    }
}
