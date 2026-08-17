import Foundation
import NVMAIDecodeProtocol
import Synchronization

/// Routes frames from the decode service socket to waiting requestors.
/// A background task reads frames and deposits them in a Mutex-protected
/// dictionary. waiters use a DispatchSemaphore to block until events arrive.
/// unchecked-invariant: the per-request waiter table is guarded by its lock.
/// The socket reader delivers replies while generation tasks register and
/// await them, so both sides contend on the same table.
final class DecodeServiceResponseRouter: @unchecked Sendable {
    private struct State {
        var pending: [UUID: [DecodeServiceEvent]] = [:]
        var terminalError: Error?
        var waiters: [UUID: DispatchSemaphore] = [:]
    }

    private let state = Mutex(State())
    private let output: FileHandle

    init(output: FileHandle) {
        self.output = output
        Task.detached(priority: .userInitiated) { [weak self] in
            await self?.readFramesInBackground()
        }
    }

    func next(matching requestID: UUID, timeout: TimeInterval) async throws
        -> DecodeServiceEvent {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<DecodeServiceEvent, Error>) in
            Task.detached(priority: .userInitiated) { [weak self] in
                guard let self else {
                    continuation.resume(throwing: DecodeFrameError.unexpectedEOF)
                    return
                }
                do {
                    let event = try self.waitForEvent(matching: requestID, timeout: timeout)
                    continuation.resume(returning: event)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private nonisolated func readFramesInBackground() async {
        do {
            while true {
                try Task.checkCancellation()
                let event = try DecodeFrameCodec.read(DecodeServiceEvent.self, from: output)
                let semaphore = state.withLock { current -> DispatchSemaphore? in
                    current.pending[event.generationID, default: []].append(event)
                    return current.waiters[event.generationID]
                }
                semaphore?.signal()
            }
        } catch is CancellationError {
            // Expected on cancellation
        } catch {
            let waiters = state.withLock { current -> [DispatchSemaphore] in
                current.terminalError = error
                return Array(current.waiters.values)
            }
            for semaphore in waiters { semaphore.signal() }
        }
        // The reader thread owns the file handle: close it here, on the
        // reader, after the blocking read returned (EOF or error) — never from
        // deinit while a read may be in flight (D18).
        output.closeFile()
    }

    private func waitForEvent(matching requestID: UUID,
                              timeout: TimeInterval) throws -> DecodeServiceEvent {
        // Register a per-request semaphore BEFORE the first dequeue so a frame
        // landing between the check and the wait cannot be lost.
        let semaphore = DispatchSemaphore(value: 0)
        state.withLock { $0.waiters[requestID] = semaphore }
        defer { state.withLock { $0.waiters[requestID] = nil } }

        if let event = dequeueEvent(for: requestID) {
            return event
        }

        let deadline = DispatchTime.now() + timeout
        while true {
            let (hasEvent, hasError) = state.withLock {
                ($0.pending[requestID]?.isEmpty == false, $0.terminalError != nil)
            }
            if hasEvent || hasError { break }
            if semaphore.wait(timeout: deadline) == .timedOut { break }
        }

        if let event = dequeueEvent(for: requestID) {
            return event
        }

        throw state.withLock({ $0.terminalError }) ?? DecodeFrameError.timedOut
    }

    private func dequeueEvent(for requestID: UUID) -> DecodeServiceEvent? {
        return state.withLock { current -> DecodeServiceEvent? in
            guard var events = current.pending[requestID], !events.isEmpty else {
                return nil
            }
            let event = events.removeFirst()
            if events.isEmpty {
                current.pending.removeValue(forKey: requestID)
            } else {
                current.pending[requestID] = events
            }
            return event
        }
    }
}
