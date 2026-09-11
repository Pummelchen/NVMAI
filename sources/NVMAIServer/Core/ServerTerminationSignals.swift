import Darwin
import Dispatch
import Foundation

public actor ServerTerminationSignals {
    // Graceful shutdown on SIGINT/SIGTERM: stop accepting work, drain, exit.
    private let stream: AsyncStream<Int32>
    private let continuation: AsyncStream<Int32>.Continuation
    private let sources: [any DispatchSourceSignal]
    /// How a second signal during shutdown ends the process.
    ///
    /// `exit(1)` in production. It is injectable because an unconditional exit
    /// is untestable in-process: the suite that exercises the second-signal path
    /// killed its own test run, and because the delivery is asynchronous it did
    /// so *after* the test returned -- aborting whichever suite ran next, with no
    /// summary and no crash report. That is the intermittent full-suite abort
    /// this register carried as unexplained.
    private let forceExit: @Sendable () -> Void

    public init(_ signals: [Int32] = [SIGINT, SIGTERM],
                forceExit: @escaping @Sendable () -> Void = { exit(1) }) {
        var capturedContinuation: AsyncStream<Int32>.Continuation?
        let stream = AsyncStream<Int32>(bufferingPolicy: .bufferingOldest(1)) {
            capturedContinuation = $0
        }
        let continuation = capturedContinuation!
        let shared = SignalState()

        self.stream = stream
        self.continuation = continuation
        self.forceExit = forceExit
        self.sources = signals.map {
            Darwin.signal($0, SIG_IGN)
            return Self.makeSource(signal: $0, continuation: continuation, state: shared,
                                   forceExit: forceExit)
        }
        for source in sources {
            source.resume()
        }
    }

    /// The signal that ended the wait, or `nil` when the wait was cancelled
    /// before any signal arrived.
    ///
    /// This used to `preconditionFailure` here ("termination signal stream ended
    /// without a signal"). Production is ordered correctly -- `main` waits and
    /// only cancels on the way out -- so the trap was unreachable rather than
    /// wrong, but nothing stops a caller of this public API from cancelling
    /// first, and a trap a caller can reach by ordering is a landmine rather
    /// than an invariant. Invariant (b) of this audit, applied to the one public
    /// ordering it could reach.
    public func wait() async -> Int32? {
        for await signal in stream {
            return signal
        }
        return nil
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
        state: SignalState,
        forceExit: @escaping @Sendable () -> Void
    ) -> any DispatchSourceSignal {
        let source = DispatchSource.makeSignalSource(signal: signal, queue: .global())
        source.setEventHandler { @Sendable [continuation, state, forceExit] in
            if state.record() {
                // S33: the first signal begins a graceful shutdown; a second
                // one during shutdown forces immediate exit instead of being
                // silently dropped. Injected so the behaviour can be asserted
                // without ending the process running the assertion.
                forceExit()
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
