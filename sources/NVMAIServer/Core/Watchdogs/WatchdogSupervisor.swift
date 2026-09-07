import Foundation

/// A `WatchdogSet` that a generation and a timer can both reach.
///
/// The stall watchdog needs a clock that ticks while nothing is happening,
/// and nothing is happening is precisely when the generation task is not
/// running. So one task streams content in and a second, tiny task calls
/// `check()` on an interval; both go through this lock.
///
/// It supervises rather than intervenes: the runner polls `wantsStop`
/// between tokens, which means a generation that has genuinely hung -- no
/// token, no poll -- is *reported* here but cannot be stopped from here.
/// That limit is real and worth stating plainly: killing a wedged GPU
/// command buffer is a process-level decision, not a watchdog's.
///
/// unchecked-invariant: every stored property is guarded by `lock`.
public final class WatchdogSupervisor: @unchecked Sendable {
    private let lock = NSLock()
    private var set: WatchdogSet
    /// Held separately from the set so the ticker can size its interval
    /// without taking the lock; it is immutable for the generation's life.
    public let configuration: WatchdogConfiguration

    public init(configuration: WatchdogConfiguration) {
        self.configuration = configuration
        set = WatchdogSet(configuration: configuration)
    }

    /// B6: the engine's own generations -- memory consolidation, above all --
    /// are not watched. Their prompts are repetitive by construction and
    /// their answers are meant to be terse, so they look like exactly the
    /// failures these detectors hunt, and no person is waiting on them.
    public static var inert: WatchdogSupervisor { WatchdogSupervisor(configuration: .off) }

    public var isActive: Bool { configuration.isEnabled }

    public var wantsStop: Bool {
        // Polled between tokens. A disabled feature must not appear on that
        // path at all, not even as an uncontended lock.
        guard isActive else { return false }
        return lock.withLock { set.wantsStop }
    }

    public var stopMessage: String? {
        lock.withLock { set.stopMessage }
    }

    public var explanation: String? {
        lock.withLock { set.explanation }
    }


    public func resolve(content: String, finishReason: String) -> WatchdogSet.Outcome {
        lock.withLock { set.resolve(content: content, finishReason: finishReason) }
    }

    public var trips: [WatchdogSet.Trip] {
        lock.withLock { set.trips }
    }

    public func observe(_ chunk: String, at instant: ContinuousClock.Instant = .now) {
        guard isActive else { return }
        lock.withLock { set.observe(chunk, at: instant) }
    }

    public func check(at instant: ContinuousClock.Instant = .now) {
        lock.withLock { set.check(at: instant) }
    }

    public func finish(visibleBytes: Int, requestBytes: Int, finishReason: String) {
        lock.withLock {
            set.finish(visibleBytes: visibleBytes, requestBytes: requestBytes,
                       finishReason: finishReason)
        }
    }

    public func record(pingPong verdict: WatchdogVerdict) {
        lock.withLock { set.record(pingPong: verdict) }
    }

    /// A task that calls `check` until the generation ends. Nil when
    /// watchdogs are off, so a feature nobody enabled starts no tasks.
    ///
    /// The interval is a fraction of the stall threshold rather than the
    /// threshold itself, so a stall is reported near when it crosses instead
    /// of up to a whole threshold late.
    public func startTicker() -> Task<Void, Never>? {
        guard isActive else { return nil }
        let interval = Duration.seconds(max(1, configuration.stallSeconds / 10))
        return Task.detached(priority: .utility) { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                guard let self, !Task.isCancelled else { return }
                self.check()
            }
        }
    }
}
