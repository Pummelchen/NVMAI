import Foundation
import Metal

/// GPU dependency for one storage batch. Value zero is never issued, and a
/// value is never reused for the lifetime of the coordinator.
/// unchecked-invariant: token fields are immutable Metal references and values.
public struct ExpertIOCompletionToken: @unchecked Sendable {
    public let event: MTLSharedEvent
    public let value: UInt64
    /// 0 = loading, 1 = complete, 2 = failed. GPU consumers wait on `event`
    /// before reading this shared word.
    public let status: MTLBuffer
    public let statusOffset: Int
}

/// Owns one shared Metal timeline for all expert reads in a model.
///
/// Out-of-order operations are deliberately held until every preceding value
/// is terminal. Advancing a timeline directly from N to N+2 would also release
/// a GPU wait for N+1, potentially before that batch's bytes were valid.
/// unchecked-invariant: allocation, status publication, and timeline advance
/// are serialized by `lock`.
public final class ExpertIOEventCoordinator: @unchecked Sendable {
    private let device: MTLDevice
    private let event: MTLSharedEvent
    private let lock = NSLock()
    private var nextValue: UInt64 = 1
    private var publishedValue: UInt64 = 0
    private var terminalValues: Set<UInt64> = []
    private var statusChunks: [MTLBuffer] = []
    private static let statusesPerChunk = 4_096
    private static let maximumStatusChunks = 4_096

    init?(device: MTLDevice) {
        guard let event = device.makeSharedEvent() else { return nil }
        event.label = "NVMAI expert I/O completion"
        self.device = device
        self.event = event
    }

    func reserve() throws -> ExpertIOCompletionToken {
        lock.lock()
        defer { lock.unlock() }
        guard nextValue != 0 && nextValue < UInt64.max else {
            throw ModelError.internalInconsistency(
                detail: "expert I/O shared-event value space exhausted")
        }
        let zeroBasedValue = Int(nextValue - 1)
        let chunkIndex = zeroBasedValue / Self.statusesPerChunk
        guard chunkIndex < Self.maximumStatusChunks else {
            throw ModelError.internalInconsistency(
                detail: "expert I/O status timeline exhausted")
        }
        if chunkIndex == statusChunks.count {
            guard let chunk = device.makeBuffer(
                length: Self.statusesPerChunk * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            statusChunks.append(chunk)
        }
        let status = statusChunks[chunkIndex]
        let statusOffset = (zeroBasedValue % Self.statusesPerChunk)
            * MemoryLayout<UInt32>.stride
        status.contents().advanced(by: statusOffset)
            .storeBytes(of: UInt32(0), as: UInt32.self)
        let token = ExpertIOCompletionToken(
            event: event,
            value: nextValue,
            status: status,
            statusOffset: statusOffset)
        nextValue &+= 1
        return token
    }

    func publish(_ token: ExpertIOCompletionToken, succeeded: Bool) {
        lock.lock()
        token.status.contents().advanced(by: token.statusOffset).storeBytes(
            of: succeeded ? UInt32(1) : UInt32(2),
            as: UInt32.self)
        terminalValues.insert(token.value)
        while terminalValues.remove(publishedValue &+ 1) != nil {
            publishedValue &+= 1
        }
        // A CPU signal after the slot bytes and status word are written is the
        // release operation paired with encodeWaitForEvent on the GPU queue.
        event.signaledValue = publishedValue
        lock.unlock()
    }

    /// Records a value already signalled by an MTLIO command buffer. Decode
    /// submits at most one such batch at a time, so values reach this shared
    /// timeline in order; keeping the same terminal set also protects future
    /// callers from accidentally advancing over an unrecorded value.
    func recordBackendSignal(_ token: ExpertIOCompletionToken) {
        lock.lock()
        terminalValues.insert(token.value)
        while terminalValues.remove(publishedValue &+ 1) != nil {
            publishedValue &+= 1
        }
        lock.unlock()
    }
}
