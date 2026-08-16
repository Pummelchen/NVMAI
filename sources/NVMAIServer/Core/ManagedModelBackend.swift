import Foundation
import NVMAI

/// Manages when the model is resident: loads it on the first inference request
/// rather than at startup, and optionally releases it again after an idle
/// period. Reloads are transparent to the caller.
///
/// The server otherwise maps ~5 GB of weights at startup and holds them for the
/// process lifetime, which on a 24 GB machine leaves no room for a second
/// workload. `ServerInferenceBackend` has two members, so residency is managed
/// by wrapping it — `NVMAIHTTPServer` needs no knowledge of any of this.
///
/// Memory is genuinely returned because every large allocation is owned by a
/// refcounted class reachable only from the session: the resident weights are
/// an `mmap` behind an `MTLBuffer` whose deallocator calls `munmap`
/// (`ResidentBuffer`), and the routed-expert slots are `posix_memalign` blocks
/// with a `free` deallocator (`PreadExpertStreamer`). Releasing the session
/// unwinds all of it.
///
/// `MetalContext` is the deliberate exception: it is built once and reused for
/// the process lifetime. `MetalContext.deinit` documents that `MTLCommandQueue`
/// has no deinit-safe cleanup, so tearing one down per unload would be unsafe;
/// it also compiles the whole shader library, which would make every reload pay
/// for a rebuild. It is small next to the weights.
public actor ManagedModelBackend: ServerInferenceBackend {
    /// Builds a session. Injectable so the residency logic can be tested
    /// against a stub without a model on disk.
    public typealias Loader =
        @Sendable (ModelSessionPlan, MetalContext?) async throws -> any ServerInferenceBackend

    /// What the reaper should do next. Returning a duration rather than
    /// polling on a fixed tick means one wake-up per idle period instead of
    /// one every few seconds, and the unload lands on the deadline instead of
    /// up to a tick late.
    enum ReaperStep: Equatable {
        case sleep(Duration)
        case stop
    }

    private let plan: ModelSessionPlan
    private let loader: Loader
    private let idleTimeout: Duration?

    /// Answered from configuration and the manifest, so they are valid before
    /// the first load and identical to what the eager path would report.
    public nonisolated let maximumContext: Int
    public nonisolated let facts: ModelSessionFacts

    private var session: (any ServerInferenceBackend)?
    /// In-flight load, so concurrent first requests coalesce into one load.
    private var loadTask: Task<any ServerInferenceBackend, any Error>?
    /// Requests currently inside `generate`. Non-zero blocks unloading.
    private var inFlight = 0
    private var lastActivity = ContinuousClock.now
    private var reaper: Task<Void, Never>?
    /// Built on the first load, then reused for the process lifetime.
    private var metalContext: MetalContext?

    public init(plan: ModelSessionPlan,
                facts: ModelSessionFacts,
                idleTimeout: Duration?,
                loader: @escaping Loader = { plan, context in
                    try await plan.makeSession(reusingContext: context)
                }) {
        self.plan = plan
        self.facts = facts
        self.idleTimeout = idleTimeout
        self.maximumContext = plan.maxContext
        self.loader = loader
    }

    // MARK: - ServerInferenceBackend

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let active = try await acquire()
        defer { release() }
        return try await active.generate(request, onEvent: onEvent)
    }

    // MARK: - Residency

    /// Marks the request in-flight *before* the first suspension point, so the
    /// reaper can never see a zero count while a request is starting up.
    private func acquire() async throws -> any ServerInferenceBackend {
        inFlight += 1
        lastActivity = .now
        do {
            return try await residentSession()
        } catch {
            inFlight -= 1
            throw error
        }
    }

    private func release() {
        inFlight -= 1
        // Measured from when a request finished, so a long generation does not
        // count against the idle window.
        lastActivity = .now
    }

    private func residentSession() async throws -> any ServerInferenceBackend {
        if let session { return session }
        if let loadTask { return try await loadTask.value }

        // Build the Metal context once; every later reload reuses it.
        let context: MetalContext
        if let metalContext {
            context = metalContext
        } else {
            context = try MetalContext()
            metalContext = context
        }

        let plan = self.plan
        let loader = self.loader
        let task = Task { try await loader(plan, context) }
        loadTask = task
        defer { loadTask = nil }

        // On failure `session` stays nil, so the next request retries cleanly
        // rather than inheriting a half-built session.
        let loaded = try await task.value
        session = loaded
        lastActivity = .now
        startReaper()
        ServerLog.residency("loaded")
        return loaded
    }

    /// Decides the reaper's next action. Split out so the policy is testable
    /// without waiting on real time.
    func reaperStep(now: ContinuousClock.Instant = .now) -> ReaperStep {
        guard let idleTimeout, session != nil else { return .stop }
        // A request in flight or a load in progress: the idle clock is not
        // running, so check again no sooner than a full timeout from now.
        guard inFlight == 0, loadTask == nil else { return .sleep(idleTimeout) }
        let idleFor = now - lastActivity
        guard idleFor >= idleTimeout else { return .sleep(idleTimeout - idleFor) }
        return .stop
    }

    private func startReaper() {
        guard idleTimeout != nil, reaper == nil else { return }
        reaper = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                switch await self.reaperStep() {
                case .sleep(let duration):
                    // A cancelled sleep exits the loop on the next check.
                    try? await Task.sleep(for: duration)
                case .stop:
                    await self.unloadIfIdle()
                    return
                }
            }
        }
    }

    /// Releases the session if it is still idle. Re-checks under actor
    /// isolation because the reaper's decision was made before its sleep.
    private func unloadIfIdle() {
        reaper = nil
        guard case .stop = reaperStep(), session != nil else {
            // Activity arrived while the reaper was waking; keep the model and
            // restart the countdown.
            startReaper()
            return
        }
        session = nil          // sole strong reference — the weights go here
        ServerLog.residency("unloaded")
    }

    // MARK: - Lifecycle

    /// Stops the reaper and drops the session. Call after the server has shut
    /// down, or the reaper outlives the process's useful life.
    public func shutdown() {
        reaper?.cancel()
        reaper = nil
        session = nil
    }

    // MARK: - Test hooks

    var isLoaded: Bool { session != nil }
    var inFlightCount: Int { inFlight }
    var hasReaper: Bool { reaper != nil }

    /// Pretend the last activity happened `duration` ago, so idle policy can be
    /// tested without sleeping.
    func backdateActivity(by duration: Duration) {
        lastActivity = .now - duration
    }

    func unloadNow() {
        session = nil
    }

    /// Simulates a request sitting inside `generate` while the reaper wakes.
    func bumpInFlightForTesting(_ delta: Int) {
        inFlight += delta
    }
}
