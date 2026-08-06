import Foundation
import NVMAIDecodeProtocol
import Synchronization

/// Routes frames from the decode service socket to waiting requestors.
/// A background task reads frames and deposits them in a Mutex-protected
/// dictionary. waiters use a DispatchSemaphore to block until events arrive.
final class DecodeServiceResponseRouter: @unchecked Sendable {
    private struct State {
        var pending: [UUID: [DecodeServiceEvent]] = [:]
        var terminalError: Error?
    }

    private let state = Mutex(State())
    private let output: FileHandle
    private let eventSignal = DispatchSemaphore(value: 0)

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
                state.withLock { $0.pending[event.generationID, default: []].append(event) }
                eventSignal.signal()
            }
        } catch is CancellationError {
            // Expected on cancellation
        } catch {
            state.withLock { $0.terminalError = error }
            eventSignal.signal()
        }
        // The reader thread owns the file handle: close it here, on the
        // reader, after the blocking read returned (EOF or error) — never from
        // deinit while a read may be in flight (D18).
        output.closeFile()
    }

    private func waitForEvent(matching requestID: UUID,
                              timeout: TimeInterval) throws -> DecodeServiceEvent {
        // First check: event already available?
        if let event = dequeueEvent(for: requestID) {
            return event
        }

        // Block until an event arrives, a terminal error occurs, or the
        // timeout elapses. The loop continues while pending[requestID] is
        // empty AND no error exists AND the deadline has not passed.
        let deadline = DispatchTime.now() + timeout
        while state.withLock({ $0.pending[requestID]?.isEmpty != false }),
              state.withLock({ $0.terminalError == nil }) {
            if eventSignal.wait(timeout: deadline) == .timedOut {
                break
            }
        }

        // Try dequeue again after signal
        if let event = dequeueEvent(for: requestID) {
            return event
        }

        // No event — throw terminal error, or a clear timeout error.
        throw state.withLock({ $0.terminalError }) ?? DecodeFrameError.timedOut
    }

    private func dequeueEvent(for requestID: UUID) -> DecodeServiceEvent? {
        guard var events = state.withLock({ $0.pending[requestID] }),
              !events.isEmpty else {
            return nil
        }
        let event = events.removeFirst()
        state.withLock { current in
            if events.isEmpty {
                current.pending.removeValue(forKey: requestID)
            } else {
                current.pending[requestID] = events
            }
        }
        return event
    }
}
