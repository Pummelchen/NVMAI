import Darwin
import Foundation

/// unchecked-invariant: the peak counter is guarded by its lock. Sampling is
/// called from the generation task while the UI reads the peak, so both sides
/// contend.
public final class AppMemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    // D30: distinguishes "no sample yet" (peakBytes == nil) from a sampled
    // zero peak (peakBytes == 0).
    private var hasPeakSample = false
    private let processFootprint: @Sendable () -> UInt64?

    public init() {
        self.processFootprint = { Self.readProcessFootprint() }
    }

    init(processFootprint: @escaping @Sendable () -> UInt64?) {
        self.processFootprint = processFootprint
    }

    public func resetPeak() {
        lock.lock()
        peak = 0
        hasPeakSample = false
        lock.unlock()
    }

    public func sample() -> UInt64? {
        guard let current = processFootprint() else { return nil }
        lock.lock()
        if !hasPeakSample || current > peak {
            peak = current
            hasPeakSample = true
        }
        lock.unlock()
        return current
    }

    private static func readProcessFootprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
    }

    public var peakBytes: UInt64? {
        lock.lock()
        let sampled = hasPeakSample
        let value = peak
        lock.unlock()
        return sampled ? value : nil
    }
}
