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

    deinit {
        output.closeFile()
    }

    func next(matching requestID: UUID) async throws -> DecodeServiceEvent {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<DecodeServiceEvent, Error>) in
            Task.detached(priority: .userInitiated) { [weak self] in
                guard let self else {
                    continuation.resume(throwing: DecodeFrameError.unexpectedEOF)
                    return
                }
                do {
                    let event = try self.waitForEvent(matching: requestID)
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
    }

    private func waitForEvent(matching requestID: UUID) throws -> DecodeServiceEvent {
        // First check: event already available?
        if let event = dequeueEvent(for: requestID) {
            return event
        }

        // Block until an event arrives or an error occurs.
        // The loop continues while pending[requestID] is empty AND no error exists.
        while state.withLock({ $0.pending[requestID]?.isEmpty != false }),
              state.withLock({ $0.terminalError == nil }) {
            _ = eventSignal.wait(timeout: .distantFuture)
        }

        // Try dequeue again after signal
        if let event = dequeueEvent(for: requestID) {
            return event
        }

        // No event — throw terminal error or unexpectedEOF
        throw state.withLock({ $0.terminalError }) ?? DecodeFrameError.unexpectedEOF
    }

    private func dequeueEvent(for requestID: UUID) -> DecodeServiceEvent? {
        if var events = state.withLock({ $0.pending[requestID] }), !events.isEmpty {
            let event = events.removeFirst()
            if events.isEmpty {
                _ = state.withLock { $0.pending.removeValue(forKey: requestID); true }
            } else {
                _ = state.withLock { $0.pending[requestID] = events; true }
            }
            return event
        }
        return nil
    }
}