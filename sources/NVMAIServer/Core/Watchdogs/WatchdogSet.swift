import Foundation

/// The one place a watchdog's report becomes a decision.
///
/// Detectors observe; this decides. A concern is logged always and stops the
/// generation only when its kind is named in `NVMAI_WATCHDOG_ACT`, so the
/// blast radius of a mis-calibrated threshold is a log line until an
/// operator deliberately widens it.
///
/// Three further rules, each of which has a test:
///
///   * **Off is free.** With watchdogs disabled nothing is allocated and no
///     detector is consulted, so a feature nobody asked for costs nothing on
///     the per-chunk path.
///   * **One report per kind.** A loop reports once, not once per byte.
///   * **Never fail a completion.** There is no throwing path here at all;
///     the worst a watchdog can do is end a generation early and say so.
public struct WatchdogSet: Sendable {
    public struct Trip: Sendable, Equatable {
        public let kind: WatchdogKind
        public let message: String
        public let acted: Bool
    }

    public let configuration: WatchdogConfiguration
    private var loop: LoopWatchdog
    private var stall: StallWatchdog
    private var stub: StubWatchdog
    public private(set) var trips: [Trip] = []
    /// Set when a watchdog that is allowed to act has tripped mid-stream.
    public private(set) var stopMessage: String?


    public init(configuration: WatchdogConfiguration) {
        self.configuration = configuration
        loop = LoopWatchdog(configuration: configuration)
        stall = StallWatchdog(configuration: configuration)
        stub = StubWatchdog(configuration: configuration)
    }

    /// B6: the engine's own generations are not watched. Memory
    /// consolidation is a server-internal call with a deliberately
    /// repetitive prompt and a deliberately terse answer -- exactly the
    /// shape the loop and stub detectors look for -- and the person never
    /// sees it, so stopping it would be a cost with no benefit.
    public static let inert = WatchdogSet(configuration: .off)

    public var isActive: Bool { configuration.isEnabled }

    /// True when a watchdog allowed to act has decided this generation
    /// should end.
    public var wantsStop: Bool { stopMessage != nil }

    public mutating func observe(_ chunk: String,
                                 at instant: ContinuousClock.Instant = .now) {
        guard configuration.isEnabled else { return }
        record(LoopWatchdog.kind, loop.observe(chunk, at: instant))
        record(StallWatchdog.kind, stall.observe(chunk, at: instant))
    }

    public mutating func check(at instant: ContinuousClock.Instant = .now) {
        guard configuration.isEnabled else { return }
        record(StallWatchdog.kind, stall.check(at: instant))
    }

    public mutating func finish(visibleBytes: Int, requestBytes: Int,
                                finishReason: String) {
        guard configuration.isEnabled else { return }
        record(StubWatchdog.kind,
               stub.finish(visibleBytes: visibleBytes, requestBytes: requestBytes,
                           finishReason: finishReason))
    }

    /// A ping-pong report from the incoming request (B2), folded in so
    /// every watchdog result reaches the log by one path.
    public mutating func record(pingPong verdict: WatchdogVerdict) {
        guard configuration.isEnabled, let message = verdict.message else { return }
        // `acted` is always false: ping-pong has no safe intervention, and
        // `WatchdogKind.canAct` records why.
        trips.append(Trip(kind: PingPongWatchdog.kind, message: message, acted: false))
    }

    private mutating func record(_ kind: WatchdogKind, _ verdict: WatchdogVerdict) {
        guard let message = verdict.message else { return }
        let acts = configuration.acts(kind)
        trips.append(Trip(kind: kind, message: message, acted: acts))
        if acts, stopMessage == nil {
            stopMessage = "\(kind.rawValue): \(message)"
        }
    }

    /// What the client is told, once the generation is over.
    ///
    /// This is the whole of the user-visible policy, in one place so it can
    /// be read and tested without a model: the note is appended to the
    /// content and the finish reason is mapped. Nothing else about the
    /// completion changes.
    public struct Outcome: Sendable, Equatable {
        public let content: String
        public let finishReason: String
        /// The text appended, or nil when nothing was.
        public let note: String?
    }

    public func resolve(content: String, finishReason: String) -> Outcome {
        guard let explanation else {
            return Outcome(content: content, finishReason: finishReason, note: nil)
        }
        // B4: `length` is the nearest honest reason either protocol offers.
        let reason = stopMessage == nil ? finishReason : "length"
        return Outcome(content: content + explanation,
                       finishReason: reason,
                       note: explanation)
    }

    /// B4: a stopped generation must say so in its content. Neither the
    /// OpenAI nor the Anthropic protocol has an honest finish reason for
    /// "the server stopped this", and inventing one breaks clients -- so the
    /// reason is mapped to the nearest existing value and the truth is told
    /// in the one place that cannot break a client, the text itself.
    public var explanation: String? {
        var notes: [String] = []
        if let stopMessage {
            notes.append("[NVMAI stopped this generation: \(stopMessage).]")
        }
        guard !notes.isEmpty else { return nil }
        return "\n\n" + notes.joined(separator: " ")
    }
}
