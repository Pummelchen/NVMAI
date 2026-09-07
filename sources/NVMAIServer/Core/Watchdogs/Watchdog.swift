import Foundation

/// Cheap, deterministic checks on a generation that has gone wrong.
///
/// Every failure these look for has been produced by this project's own
/// runs -- a 0.8B looping on "Wait, I need to check the facts again" until
/// its token budget died, a bound port that never produced a token, a
/// forty-token reply where a program belonged, a tool called with the same
/// arguments until the rounds ran out. None of them needs a model to spot,
/// which is the whole reason they are kept separate from the memory work:
/// no second generation, no extra memory, no dependence on a model's
/// judgement.
///
/// The design rule that matters is that a **detector never decides to stop**.
/// It reports what it saw, and `WatchdogSet` alone turns a report into a
/// stop, and only for the kinds the operator named in `NVMAI_WATCHDOG_ACT`.
/// A false stop costs the user a whole answer, so the policy lives in one
/// place where it can be read in ten lines.
public enum WatchdogKind: String, Sendable, CaseIterable {
    case loop
    case stall
    case stub
    case pingpong

    /// Whether this watchdog has an intervention at all.
    ///
    /// `pingpong` does not, and the reason is worth recording because the
    /// obvious intervention was tried and is unsafe. The loop lives in the
    /// request that just arrived, so there is no generation to stop; the
    /// apparent answer is to answer that turn with no tools offered. But a
    /// history containing tool calls still renders with the tool template,
    /// so the model goes on emitting tool calls into a decoder that now
    /// allows none of them -- and the request fails outright, which is worse
    /// than the loop and breaks this module's own rule that a watchdog never
    /// fails a completion. Rendering the history without the tool template
    /// is not an option either: it is what the transcript is written in.
    ///
    /// So ping-pong reports, and the client, which owns the loop, decides.
    public var canAct: Bool { self != .pingpong }
}

/// What a detector has to say. `concern` is a detector's strongest verdict;
/// `stop` is only ever produced by `WatchdogSet`, from a concern about a kind
/// that is allowed to act.
public enum WatchdogVerdict: Sendable, Equatable {
    case fine
    case concern(String)
    case stop(String)

    public var message: String? {
        switch self {
        case .fine: nil
        case .concern(let text), .stop(let text): text
        }
    }
}

/// One check over a generation. Three entry points, because the failures
/// arrive at three different moments: as content streams (`observe`), while
/// nothing at all arrives (`check`), and once the generation is over
/// (`finish`). Most detectors implement one of the three.
///
/// B5: every implementation costs O(1) per chunk. A detector whose cost grew
/// with the output would tax exactly the long generations most likely to
/// need watching, and `WatchdogCostTests` asserts it does not.
public protocol Watchdog: Sendable {
    static var kind: WatchdogKind { get }
    mutating func observe(_ chunk: String, at instant: ContinuousClock.Instant) -> WatchdogVerdict
    mutating func check(at instant: ContinuousClock.Instant) -> WatchdogVerdict
    mutating func finish(visibleBytes: Int, requestBytes: Int,
                         finishReason: String) -> WatchdogVerdict
}

public extension Watchdog {
    mutating func observe(_ chunk: String,
                          at instant: ContinuousClock.Instant) -> WatchdogVerdict { .fine }
    mutating func check(at instant: ContinuousClock.Instant) -> WatchdogVerdict { .fine }
    mutating func finish(visibleBytes: Int, requestBytes: Int,
                         finishReason: String) -> WatchdogVerdict { .fine }
}

/// The whole configuration surface. Off by default, and observation-only
/// even when on: a watchdog may stop a generation only when its name appears
/// in `NVMAI_WATCHDOG_ACT`.
public struct WatchdogConfiguration: Sendable, Equatable {
    public var isEnabled: Bool
    /// Kinds allowed to stop a generation. Empty is observation only.
    public var acting: Set<WatchdogKind>
    /// Seconds without a visible token, measured from the **first** token
    /// (B1). Prefill emits nothing and this project has measured a 10k-token
    /// prompt taking 652 s of it, so a clock started at the request would
    /// stop every long prompt.
    public var stallSeconds: Double
    /// Repeats of one window before a loop is called. Six, measured: see
    /// `loopWindowBytes` for the calibration this came from.
    public var loopRepeats: Int
    /// Length of the repeated window, in bytes.
    ///
    /// Sixty-four and six repeats sit in the middle of a plateau measured
    /// over 999 recorded replies (`benchmark/watchdog_calibrate.py`): every
    /// combination from a 56-byte window at five repeats upwards produced
    /// zero false positives, and all of them still caught the one genuine
    /// loop in the corpus, a C99 reply that emitted the same
    /// `SDL_SetRenderDrawColor` line over and over. The plan's proposed 40
    /// bytes at four repeats fired on 8.1% of the corpus, all of it real
    /// code -- repeated SDL calls, repeated struct initialisers -- which is
    /// exactly the false-positive tail B3 predicted.
    public var loopWindowBytes: Int
    /// How far back a repeat still counts, in bytes.
    public var loopHistoryBytes: Int
    /// A reply that finished normally with fewer visible bytes than this is
    /// a stub. Roughly 24 tokens at four bytes a token; see `StubWatchdog`
    /// for why the rule is written in bytes rather than tokens.
    public var stubVisibleBytes: Int
    /// Bytes the last user message must reach before a short reply counts as
    /// a stub.
    ///
    /// Without this the rule fires on "Say OK." answered with "OK.", which
    /// is exactly what it did on the first request of the first observation
    /// run: the harness's readiness probe is that shape. A short answer to a
    /// short question is an answer, and no rule that ignores the question
    /// can tell the two apart. Two hundred bytes separates the recorded
    /// corpus cleanly -- the probe is 7 bytes, every real book request is
    /// 657 or more -- and over 289 recorded exchanges it flags none.
    public var stubAskedBytes: Int
    /// Identical tool calls in one request's history before it is a loop.
    public var pingPongRepeats: Int

    public init(isEnabled: Bool = false,
                acting: Set<WatchdogKind> = [],
                stallSeconds: Double = 90,
                loopRepeats: Int = 6,
                loopWindowBytes: Int = 64,
                loopHistoryBytes: Int = 1_200,
                stubVisibleBytes: Int = 96,
                stubAskedBytes: Int = 200,
                pingPongRepeats: Int = 3) {
        self.isEnabled = isEnabled
        self.acting = acting
        self.stallSeconds = max(1, stallSeconds)
        self.loopRepeats = max(2, loopRepeats)
        self.loopWindowBytes = max(16, loopWindowBytes)
        // Against the clamped window, not the parameter: otherwise a small
        // window argument leaves a history shorter than the window it is
        // supposed to contain.
        self.loopHistoryBytes = max(self.loopWindowBytes * 2, loopHistoryBytes)
        self.stubVisibleBytes = max(0, stubVisibleBytes)
        self.stubAskedBytes = max(0, stubAskedBytes)
        self.pingPongRepeats = max(2, pingPongRepeats)
    }

    public static let off = WatchdogConfiguration()

    public func acts(_ kind: WatchdogKind) -> Bool {
        isEnabled && kind.canAct && acting.contains(kind)
    }

    /// Read once at startup and held as a static. Per-call
    /// `ProcessInfo.environment` reads have already cost this project about
    /// 40% of a 35B token once; a flag on the per-chunk path must never be
    /// read from the environment.
    public static let shared = fromEnvironment()

    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> WatchdogConfiguration {
        var configuration = WatchdogConfiguration()
        let flag = environment["NVMAI_WATCHDOGS"]?.lowercased()
        configuration.isEnabled = flag == "1" || flag == "on" || flag == "true"
        if let list = environment["NVMAI_WATCHDOG_ACT"] {
            configuration.acting = Set(list
                .split(separator: ",")
                .compactMap { WatchdogKind(rawValue: $0.trimmingCharacters(in: .whitespaces)
                    .lowercased()) }
                .filter(\.canAct))
        }
        if let value = environment["NVMAI_WATCHDOG_STALL_SECONDS"].flatMap(Double.init) {
            configuration.stallSeconds = max(1, value)
        }
        if let value = environment["NVMAI_WATCHDOG_LOOP_REPEATS"].flatMap(Int.init) {
            configuration.loopRepeats = max(2, value)
        }
        return configuration
    }

    /// One word for the startup banner.
    public var summary: String {
        guard isEnabled else { return "watchdogs=off" }
        guard !acting.isEmpty else { return "watchdogs=observe" }
        let names = WatchdogKind.allCases
            .filter(acting.contains)
            .map(\.rawValue)
            .joined(separator: ",")
        return "watchdogs=act(\(names))"
    }
}


public extension WatchdogConfiguration {
    /// Say what is watching, at startup, on the channel a server log
    /// actually captures.
    func announce() {
        ServerLog.watchdogStartup(summary)
    }
}
