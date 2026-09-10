import Foundation
import NVMAI

/// What the HTTP layer needs to validate a request for one model before that
/// model is resident: the omitted-sampling defaults, the max_tokens bound and
/// the reasoning profile all belong to the model a request names, not to
/// whichever one happens to be loaded.
public struct ServedModel: Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let maximumContext: Int
    public let sampling: GenerationDefaults.Sampling
    public let reasoningProfile: ServerReasoningProfile

    public init(id: String, displayName: String, maximumContext: Int,
                sampling: GenerationDefaults.Sampling,
                reasoningProfile: ServerReasoningProfile) {
        self.id = id
        self.displayName = displayName
        self.maximumContext = maximumContext
        self.sampling = sampling
        self.reasoningProfile = reasoningProfile
    }
}

/// A backend that serves several models by name. Kept apart from
/// `ServerInferenceBackend` for the reason `ResidencyManaging` is: one backend
/// routes, and every other conformer would carry a member that answers "not me".
public protocol ModelRouting: Sendable {
    /// Every model a request may name, in listing order.
    var servedModels: [ServedModel] { get }
}

public extension ModelRouting {
    /// The model a request's `model` field names, or nil for an unknown name.
    /// An exact id wins over the "<model>-fast" alias, so a catalog id that
    /// itself ends in "-fast" stays reachable.
    func servedModel(named name: String) -> ServedModel? {
        if let exact = servedModels.first(where: { $0.id == name }) { return exact }
        guard name.hasSuffix("-fast") else { return nil }
        let base = String(name.dropLast("-fast".count))
        return servedModels.first { $0.id == base }
    }
}

/// The reasoning one model runs under for the server-wide level.
public struct ReasoningChoice: Sendable, Equatable {
    public let requested: ReasoningLevel
    public let effective: ReasoningLevel
    public let thinking: ModelThinkingMode
    public let effort: ModelReasoningEffort?
}

/// Fits one server-wide reasoning level to models that expose different ones.
///
/// The level is chosen once for the server and the models under it differ:
/// Qwen 3.6 has an on/off switch, Qwen3.8-Flash-Next has effort levels and no
/// bare "on". Refusing to load a model because the level does not map exactly
/// would make `--reasoning` useless with a mixed catalog, so each model gets
/// the closest thing its template defines.
public enum ReasoningFallback {
    /// `whenOn` is what the model's template does when thinking is switched
    /// on with no effort named; `ModelCatalog.Kind.levelWhenOn` supplies it.
    public static func effectiveLevel(_ requested: ReasoningLevel,
                                      supported: [ReasoningLevel],
                                      whenOn: ReasoningLevel? = nil) -> ReasoningLevel {
        if supported.contains(requested) || requested == .off { return requested }
        let efforts = supported.filter { $0 != .off && $0 != .on }
        // An on/off model: any effort means "think".
        guard !efforts.isEmpty else { return supported.contains(.on) ? .on : .off }
        // "On" for an effort model: the template's own default, extra high
        // for Qwen3.8. That is what --thinking on has always loaded on a
        // single-model server, and the same flag must not think less because
        // the server was started with a catalog. The middle effort is left
        // only for a caller that cannot say what the template does.
        if requested == .on {
            if let whenOn, efforts.contains(whenOn) { return whenOn }
            return efforts[(efforts.count - 1) / 2]
        }
        // An effort the model lacks: the nearest one it has, ties to the
        // cheaper, since a client that wanted more can ask for it by name.
        let order = ReasoningLevel.allCases
        let rank = { (level: ReasoningLevel) in order.firstIndex(of: level) ?? 0 }
        let target = rank(requested)
        return efforts.min { lhs, rhs in
            let left = abs(rank(lhs) - target), right = abs(rank(rhs) - target)
            return left == right ? rank(lhs) < rank(rhs) : left < right
        } ?? .on
    }

    public static func choice(for kind: ModelCatalog.Kind,
                              requested: ReasoningLevel) throws -> ReasoningChoice {
        let effective = effectiveLevel(requested, supported: kind.supportedReasoningLevels,
                                       whenOn: kind.levelWhenOn)
        let runtime = try kind.runtimeReasoning(for: effective)
        return ReasoningChoice(requested: requested, effective: effective,
                               thinking: runtime.thinking, effort: runtime.effort)
    }
}

public enum ModelRouterError: Error, CustomStringConvertible {
    case notInCatalog(String)

    public var description: String {
        switch self {
        case .notInCatalog(let model):
            "\(model) is not a model the catalog can serve; run with --catalog to list them"
        }
    }
}

/// Serves every catalog model through one server, keeping at most one resident.
///
/// A request names its model; if that is not the resident one, the router
/// waits for in-flight work on the resident model to drain, releases it, and
/// loads the requested one. It never holds two: on a 26 GB machine two 35B
/// models would swap, so the old weights are dropped before the new ones are
/// mapped, and a switch costs a full load.
///
/// The HTTP coordinator already runs one generation at a time, so a switch
/// normally happens inside that serialised section with nothing else running.
/// The router does not rely on it: token counting and memory consolidation
/// reach the backend outside the coordinator, so residency is guarded here by
/// an in-flight count, exactly as `ManagedModelBackend` guards its unloads.
public actor ModelRouter: ServerInferenceBackend, ResidencyManaging, PromptTokenCounting, ModelRouting {
    /// Builds a backend for one catalog entry. Injectable so switching can be
    /// tested against stubs without a model on disk.
    public typealias Loader =
        @Sendable (ModelCatalog.Entry, ReasoningChoice) async throws -> any ServerInferenceBackend
    /// Counts a request's prompt tokens for a model that is not resident,
    /// from its tokenizer alone. Injectable for the same reason.
    public typealias Counter =
        @Sendable (ModelCatalog.Entry, ReasoningChoice, ValidatedChatRequest) async throws -> Int

    public nonisolated let servedModels: [ServedModel]
    public nonisolated let initialModelID: String
    private nonisolated let initial: ServedModel
    private nonisolated let entries: [String: ModelCatalog.Entry]
    private nonisolated let choices: [String: ReasoningChoice]
    private let loader: Loader
    private let counter: Counter

    /// The protocol's single-model view, answered for the model loaded first.
    /// The HTTP layer asks `servedModel(named:)` instead once it routes.
    public nonisolated var maximumContext: Int { initial.maximumContext }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { initial.sampling }

    private struct Resident {
        let id: String
        let backend: any ServerInferenceBackend
    }

    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Never>
    }

    private var resident: Resident?
    /// Callers currently using the resident backend. Non-zero blocks a switch.
    private var inFlight = 0
    /// A load is running; nothing is resident until it finishes.
    private var switching = false
    /// Callers waiting to switch away from the resident model. While any wait,
    /// new work for the resident model queues behind them rather than keeping
    /// the in-flight count above zero forever.
    private var pendingSwitches = 0
    private var waiters: [Waiter] = []

    public init(catalog: ModelCatalog,
                initialModelID: String,
                reasoning: ReasoningLevel,
                maximumContext: Int,
                loader: @escaping Loader,
                counter: @escaping Counter = ModelRouter.standardCounter) throws {
        var served: [ServedModel] = []
        var choices: [String: ReasoningChoice] = [:]
        var entries: [String: ModelCatalog.Entry] = [:]
        for entry in catalog.entries {
            let choice = try ReasoningFallback.choice(for: entry.kind, requested: reasoning)
            choices[entry.id] = choice
            entries[entry.id] = entry
            served.append(ServedModel(
                id: entry.id,
                displayName: entry.name,
                maximumContext: Self.context(for: entry, configured: maximumContext),
                sampling: entry.sampling,
                reasoningProfile: Self.profile(for: entry, choice: choice)))
        }
        guard let initial = served.first(where: { $0.id == initialModelID }) else {
            throw ModelRouterError.notInCatalog(initialModelID)
        }
        self.servedModels = served
        self.initialModelID = initialModelID
        self.initial = initial
        self.entries = entries
        self.choices = choices
        self.loader = loader
        self.counter = counter
    }

    /// Mirrors `CPUModelBackend`'s own clamp, so validation bounds max_tokens
    /// by the context the CPU engine will actually give the request.
    static func context(for entry: ModelCatalog.Entry, configured: Int) -> Int {
        switch entry.backend {
        case .gpu: configured
        case .cpu: min(configured, entry.contextLimit ?? CPUModelBackend.contextCeiling,
                       CPUModelBackend.contextCeiling)
        }
    }

    /// A CPU family's template has the binary switch Qwen 3.6 has, so it is
    /// validated as that family, as the single-model CPU path already does.
    static func profile(for entry: ModelCatalog.Entry,
                        choice: ReasoningChoice) -> ServerReasoningProfile {
        let family: ModelFamily
        switch entry.kind {
        case .gpu(let gpuFamily): family = gpuFamily
        case .cpu: family = .qwen36
        }
        return ServerReasoningProfile(family: family, thinkingMode: choice.thinking,
                                      reasoningEffort: choice.effort)
    }

    public func reasoningChoice(for id: String) -> ReasoningChoice? { choices[id] }

    // MARK: - ServerInferenceBackend

    /// A request with no model -- the engine's own, such as memory
    /// consolidation -- runs on whatever is resident rather than forcing a load.
    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let active = try await acquire(request.model)
        defer { release() }
        return try await active.generate(request, onEvent: onEvent)
    }

    /// A count is a question about text, not a reason to switch. One naming
    /// a model that is not resident is answered from that model's tokenizer
    /// and the resident model stays loaded; routing it like a generation
    /// made the next generation pay a full reload for a number.
    public func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        let target = request.model ?? resident?.id ?? initialModelID
        if target != resident?.id, let entry = entries[target], let choice = choices[target] {
            return try await counter(entry, choice, request)
        }
        let active = try await acquire(request.model)
        defer { release() }
        guard let counting = active as? any PromptTokenCounting else {
            throw ServerRequestError.unsupportedOperation("count_tokens")
        }
        return try await counting.countPromptTokens(request)
    }

    /// Loads the initial model, so a server that was not asked to defer pays
    /// the load at startup rather than on the first request.
    public func preload() async throws {
        _ = try await acquire(initialModelID)
        release()
    }

    // MARK: - ResidencyManaging

    /// Releases the resident model once in-flight work drains. The next
    /// request loads whichever model it names.
    public func unload() async -> Bool {
        while resident != nil || switching {
            if !switching, inFlight == 0, let released = resident {
                resident = nil
                ServerLog.residency("unloaded \(released.id)")
                return true
            }
            if Task.isCancelled { return false }
            await waitForTurn()
        }
        return false
    }

    public func shutdown() {
        resident = nil
    }

    public var residentModelID: String? { resident?.id }

    // MARK: - Residency

    /// Marks the caller in flight on the backend for `requested`, switching to
    /// it first when it is not resident. Every exit that does not return a
    /// backend leaves the in-flight count untouched.
    private func acquire(_ requested: String?) async throws -> any ServerInferenceBackend {
        let target = requested ?? resident?.id ?? initialModelID
        guard let entry = entries[target], let choice = choices[target] else {
            throw ServerRequestError.unknownModel
        }
        while true {
            try Task.checkCancellation()
            if let resident, resident.id == target, !switching, pendingSwitches == 0 {
                inFlight += 1
                return resident.backend
            }
            if resident?.id != target, !switching, inFlight == 0 {
                return try await switchTo(entry, choice: choice)
            }
            if resident?.id == target {
                await waitForTurn()
                continue
            }
            pendingSwitches += 1
            await waitForTurn()
            pendingSwitches -= 1
            if Task.isCancelled {
                // Work for the resident model may be queued behind this
                // switch; with it abandoned they must re-check, not sleep on.
                wakeWaiters()
                throw CancellationError()
            }
        }
    }

    private func switchTo(_ entry: ModelCatalog.Entry,
                          choice: ReasoningChoice) async throws -> any ServerInferenceBackend {
        switching = true
        defer {
            switching = false
            wakeWaiters()
        }
        if let previous = resident {
            // Dropped before the next model is mapped: this reference is the
            // last one, so the old weights are gone before the new ones load.
            resident = nil
            ServerLog.residency("unloaded \(previous.id) to load \(entry.id)")
        }
        // Unstructured, so a client that disconnects mid-load does not abort a
        // load that the requests queued behind it are waiting for.
        let loader = self.loader
        let loaded = try await Task { try await loader(entry, choice) }.value
        resident = Resident(id: entry.id, backend: loaded)
        inFlight += 1
        let fitted = choice.effective == choice.requested
            ? "" : " (server level \(choice.requested.rawValue))"
        ServerLog.residency("loaded \(entry.id) on the \(entry.backend.rawValue) "
            + "reasoning=\(choice.effective.rawValue)\(fitted)")
        return loaded
    }

    private func release() {
        inFlight -= 1
        if inFlight == 0 { wakeWaiters() }
    }

    private func waitForTurn() async {
        let id = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                // Checked here, on the actor, so a cancellation that landed
                // before this waiter was queued cannot leave it asleep.
                if Task.isCancelled {
                    continuation.resume()
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume()
    }

    /// Every waiter re-checks its own condition, so waking all of them is
    /// always safe; the ones that still cannot proceed queue again.
    private func wakeWaiters() {
        let woken = waiters
        waiters.removeAll()
        for waiter in woken {
            waiter.continuation.resume()
        }
    }

    // MARK: - Test hooks

    var inFlightCount: Int { inFlight }
    var waiterCount: Int { waiters.count }
}

extension ModelRouter {
    /// Loads entries the way the single-model paths do: a GPU install through
    /// `ModelSessionPlan` with every server flag, a CPU snapshot through
    /// `CPUModelBackend`. Reasoning comes from the router, fitted per model.
    public static func standardLoader(arguments: ServerArguments) -> Loader {
        let metal = SharedMetalContext()
        return { entry, reasoning in
            switch entry.kind {
            case .gpu:
                let plan = ModelSessionPlan(
                    modelDirectory: entry.path,
                    maxContext: arguments.maxContext,
                    promptCacheMode: arguments.promptCacheMode,
                    promptCacheMaximumEntries: arguments.promptCacheMaximumEntries,
                    promptCacheMemoryLimitBytes: arguments.promptCacheMemoryMiB * 1_048_576,
                    promptCacheDiskDirectory: arguments.promptCacheDiskDirectory.map {
                        URL(fileURLWithPath: $0).standardizedFileURL
                    },
                    promptCacheDiskLimitBytes: arguments.promptCacheDiskMiB * 1_048_576,
                    prefillChunkTokens: arguments.prefillChunkTokens,
                    kvCachePrecision: arguments.kvCachePrecision,
                    ropeScalingMode: arguments.ropeScalingMode,
                    thinkingMode: reasoning.thinking,
                    reasoningEffort: reasoning.effort,
                    expertCacheSlots: arguments.expertCacheSlots,
                    expertCacheBudgetBytes: arguments.expertCacheBudgetBytes,
                    mtpModelDirectory: nil,
                    mtpMemoryMiB: arguments.mtpMemoryMiB)
                return try await plan.makeSession(reusingContext: try await metal.context())
            case .cpu:
                return try await CPUModelBackend(
                    snapshotDirectory: entry.path,
                    maximumContext: arguments.maxContext,
                    resident: arguments.cpuResident,
                    thinkingMode: reasoning.thinking)
            }
        }
    }
}

extension ModelRouter {
    /// Counts with the named model's tokenizer, rendered as its engine
    /// renders a prompt, without loading any weights.
    public static let standardCounter: Counter = { entry, choice, request in
        switch entry.kind {
        case .gpu:
            guard let folder = GFTokenizer.tokenizerFolder(forModelDirectory: entry.path) else {
                throw GFTokenizerError.missingToolTemplate
            }
            let tokenizer = try await GFTokenizer.load(from: folder,
                                                       thinkingMode: choice.thinking,
                                                       reasoningEffort: choice.effort)
            return try ServerModelSession.promptTokenCount(request, tokenizer: tokenizer)
        case .cpu:
            let tokenizer = try await GFTokenizer.load(from: entry.path,
                                                       thinkingMode: choice.thinking)
            return try CPUModelBackend.promptTokenCount(request, tokenizer: tokenizer)
        }
    }
}

/// One Metal context for the process, built on the first GPU load. A command
/// queue has no deinit-safe teardown (see `MetalContext.deinit`) and the
/// shader library is costly to compile, so every switch reuses it.
private actor SharedMetalContext {
    private var built: MetalContext?

    func context() throws -> MetalContext {
        if let built { return built }
        let context = try MetalContext()
        built = context
        return context
    }
}

extension ServerArguments {
    /// The thinking settings a single-model server loads with. Without
    /// `--reasoning` these are the `--thinking` / `--reasoning-effort` values
    /// verbatim, as before; with it, the level is fitted to the model's family
    /// exactly as the router fits it.
    public func singleModelReasoning(
        directory: URL
    ) throws -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
        guard let reasoningLevel else { return (thinkingMode, reasoningEffort) }
        let kind: ModelCatalog.Kind = cpu
            ? .cpu(try ModelCatalog.snapshotFamily(directory))
            : .gpu(try ManifestReader.peekFamily(directoryURL: directory))
        let choice = try ReasoningFallback.choice(for: kind, requested: reasoningLevel)
        ServerLog.residency("reasoning=\(choice.effective.rawValue)"
            + (choice.effective == reasoningLevel ? "" : " (server level \(reasoningLevel.rawValue))"))
        return (choice.thinking, choice.effort)
    }
}
