import Darwin
import Dispatch
import Foundation

public actor ServerTerminationSignals {
    // Graceful shutdown on SIGINT/SIGTERM: stop accepting work, drain, exit.
    private let stream: AsyncStream<Int32>
    private let continuation: AsyncStream<Int32>.Continuation
    private let sources: [any DispatchSourceSignal]

    public init(_ signals: [Int32] = [SIGINT, SIGTERM]) {
        var capturedContinuation: AsyncStream<Int32>.Continuation?
        let stream = AsyncStream<Int32>(bufferingPolicy: .bufferingOldest(1)) {
            capturedContinuation = $0
        }
        let continuation = capturedContinuation!
        let shared = SignalState()

        self.stream = stream
        self.continuation = continuation
        self.sources = signals.map {
            Darwin.signal($0, SIG_IGN)
            return Self.makeSource(signal: $0, continuation: continuation, state: shared)
        }
        for source in sources {
            source.resume()
        }
    }

    public func wait() async -> Int32 {
        for await signal in stream {
            return signal
        }
        preconditionFailure("termination signal stream ended without a signal")
    }

    public func cancel() {
        for source in sources {
            source.cancel()
        }
        continuation.finish()
    }

    private nonisolated static func makeSource(
        signal: Int32,
        continuation: AsyncStream<Int32>.Continuation,
        state: SignalState
    ) -> any DispatchSourceSignal {
        let source = DispatchSource.makeSignalSource(signal: signal, queue: .global())
        source.setEventHandler { @Sendable [continuation, state] in
            if state.record() {
                // S33: the first signal begins a graceful shutdown; a second
                // one during shutdown forces immediate exit instead of being
                // silently dropped.
                exit(1)
            } else {
                continuation.yield(signal)
            }
        }
        return source
    }
}

/// Shared first-signal bookkeeping across the per-signal dispatch sources.
/// unchecked-invariant: `delivered` is guarded by `lock`. Signal handlers can
/// fire on any thread and may fire more than once, so the flag exists to make
/// the first delivery win and the rest no-ops.
private final class SignalState: @unchecked Sendable {
    private let lock = NSLock()
    private var delivered = false

    /// Returns true when a signal arrives after the first one.
    func record() -> Bool {
        lock.withLock {
            if delivered { return true }
            delivered = true
            return false
        }
    }
}
