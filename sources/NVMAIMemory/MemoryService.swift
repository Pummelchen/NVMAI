import Foundation
import ContinuityCore

/// What the serving engine talks to.
///
/// It owns the choice of backend, the fallback when durable storage cannot
/// be written, and the session lifecycle. The engine calls four things: `beginSession`,
/// `instructions`, `execute` and `endSession`. Everything else stays here.
///
/// The service never throws at the engine. Memory is optional by design, so
/// a failure produces a logged event and a degraded mode, not a failed
/// completion. The one thing it will not do is report a write as successful
/// when it was not.
public actor MemoryService {
    public private(set) var configuration: MemoryConfiguration
    private let durableStore: (any MemoryStore)?
    /// The in-process engine, when this service built one. Nil when a caller
    /// injected its own store, which is how the tests drive it.
    private let engine: ContinuityEngine?
    private let localStore: InMemoryStore
    /// The engine-authored journal. Separate store, separate key space,
    /// separate trim policy: a busy week of sessions must never evict the
    /// facts the model wrote deliberately.
    private let journal: (any SessionJournal)?
    private let journalFilter: JournalFilter
    /// Set once a durable operation has failed, so the session prompt can say
    /// memory is not persisting instead of the model assuming it is.
    private var isDegraded = false
    private var engineStarted = false
    /// False when the engine is running without a journal, so writes last
    /// only as long as the process.
    private var enginePersists = true
    private var log: @Sendable (MemoryLogEvent) -> Void

    public init(configuration: MemoryConfiguration,
                durableStore: (any MemoryStore)? = nil,
                journal: (any SessionJournal)? = nil,
                log: @escaping @Sendable (MemoryLogEvent) -> Void = { _ in }) {
        self.configuration = configuration
        self.localStore = InMemoryStore(limits: configuration.limits)
        self.journalFilter = configuration.journalLimits.filter
        self.log = log
        if let durableStore {
            self.durableStore = durableStore
            self.engine = nil
        } else if configuration.isEnabled {
            // One engine for both stores: the same file, the same restart,
            // and a session that means the same thing to each of them.
            let (engine, persists) = Self.makeEngine(configuration: configuration, log: log)
            let store = ContinuityStore(engine: engine, limits: configuration.limits)
            self.engine = engine
            self.durableStore = store
            self.enginePersists = persists
        } else {
            self.engine = nil
            self.durableStore = nil
        }
        if let journal {
            self.journal = journal
        } else if configuration.isEnabled, configuration.journalEnabled,
                  let engine = self.engine,
                  let store = self.durableStore as? ContinuityStore {
            self.journal = ContinuityJournalStore(engine: engine, store: store,
                                                  limits: configuration.journalLimits)
        } else {
            self.journal = nil
        }
    }

    /// Builds the engine, with a journal file when one can be opened.
    ///
    /// A directory that cannot be written is not fatal: the engine still runs
    /// in memory for the session. It is logged, and `isDurable` reports false,
    /// so the prompt tells the model its writes will not outlive the session
    /// rather than letting it assume they will.
    ///
    /// - Returns: the engine, and whether it is actually writing to a file.
    ///   The flag is not cosmetic: without it a session whose journal could
    ///   not be opened would tell the model its writes persist, which is the
    ///   one thing memory must never get wrong.
    private static func makeEngine(
        configuration: MemoryConfiguration,
        log: @Sendable (MemoryLogEvent) -> Void
    ) -> (engine: ContinuityEngine, persists: Bool) {
        let limits = ContinuityCore.MemoryLimits(
            maxValueBytes: configuration.limits.maximumValueBytes,
            maxItemsPerTask: itemCeiling(for: configuration))
        let engineConfiguration = ContinuityConfiguration(
            memoryLimits: limits,
            journalsSessionContent: configuration.journalEnabled)
        guard let scope = configuration.scope() else {
            return (ContinuityEngine(configuration: engineConfiguration), false)
        }
        do {
            let journal = try FileJournal(
                url: configuration.storage.journalURL(for: scope),
                synchronizesEveryWrite: configuration.storage.synchronizesEveryWrite)
            return (ContinuityEngine(configuration: engineConfiguration, journal: journal), true)
        } catch {
            log(.degraded(operation: "openJournal", detail: "\(error)"))
            return (ContinuityEngine(configuration: engineConfiguration), false)
        }
    }

    /// Turns the byte budget into an item ceiling.
    ///
    /// The bound is the worst case, every record at the maximum value size,
    /// so the store cannot exceed the budget the machine was sized for even
    /// when every fact the model writes is enormous.
    static func itemCeiling(for configuration: MemoryConfiguration) -> Int {
        guard let budget = configuration.storage.maximumMemoryBytes, budget > 0 else {
            return 8192
        }
        let perItem = max(1, configuration.limits.maximumValueBytes)
        return max(64, budget / perItem)
    }

    /// Records a completed turn. Content is filtered to substance here, so no
    /// caller can accidentally journal a tool result or a file dump.
    public func recordTurn(session: MemorySessionContext,
                           index: Int,
                           prompt: String,
                           reply: String,
                           model: String?,
                           promptTokens: Int,
                           completionTokens: Int,
                           latencyMilliseconds: Int,
                           stopReason: String?) async {
        guard let journal else { return }
        let filteredPrompt = journalFilter.filter(prompt)
        let filteredReply = journalFilter.filter(reply)
        let turn = JournalTurn(session: session.session.id,
                               workspace: session.scope.workspace,
                               index: index,
                               prompt: filteredPrompt.kept,
                               reply: filteredReply.kept,
                               model: model,
                               promptTokens: promptTokens,
                               completionTokens: completionTokens,
                               latencyMilliseconds: latencyMilliseconds,
                               stopReason: stopReason,
                               droppedBytes: filteredPrompt.dropped + filteredReply.dropped)
        await journal.record(turn, in: session.scope)
        log(.journaled(session: session.session.id, index: index, bytes: turn.byteCount))
    }

    /// The journal, for a caller that wants to read it back. Never used to
    /// build a prompt.
    public func journalStore() -> (any SessionJournal)? { journal }

    public var isEnabled: Bool { configuration.isEnabled }

    /// Whether writes are currently reaching durable storage. False once a
    /// durable operation has failed and the local fallback took over.
    public var isDurable: Bool { durableStore != nil && !isDegraded && enginePersists }

    /// Starts a session and returns what the engine needs to install.
    ///
    /// A failure here degrades rather than propagates: the session continues
    /// with local memory when that is allowed, and with none when it is not.
    public func beginSession(id: String,
                             workspaceOverride: String? = nil,
                             modelID: String? = nil) async -> MemorySessionContext? {
        guard configuration.isEnabled else { return nil }
        await startEngineIfNeeded()
        guard let scope = configuration.scope(workspaceOverride: workspaceOverride) else {
            log(.rejectedScope(workspaceOverride ?? configuration.workspace))
            return nil
        }
        let session = MemorySession(id: id, modelID: modelID)
        var bootstrap = MemoryBootstrap.empty
        if let durableStore {
            do {
                bootstrap = try await durableStore.sessionInit(session, in: scope)
                isDegraded = false
            } catch {
                isDegraded = true
                log(.degraded(operation: "sessionInit", detail: "\(error)"))
                guard configuration.degradesToLocalStore else { return nil }
                bootstrap = (try? await localStore.sessionInit(session, in: scope)) ?? .empty
            }
        } else {
            bootstrap = (try? await localStore.sessionInit(session, in: scope)) ?? .empty
        }
        log(.sessionStarted(session: session.id, scope: scope,
                            bootstrapRecords: bootstrap.records.count,
                            bootstrapBytes: bootstrap.totalBytes))
        return MemorySessionContext(session: session,
                                    scope: scope,
                                    bootstrap: bootstrap,
                                    isDurable: isDurable)
    }

    /// Replays the journal once, before the first session.
    ///
    /// Deferred rather than done in `init` because a replay is I/O and an
    /// initializer that reads a file cannot report a failure to the caller
    /// that will actually be affected by it.
    private func startEngineIfNeeded() async {
        guard let engine, !engineStarted else { return }
        engineStarted = true
        do {
            try await engine.start()
        } catch {
            isDegraded = true
            log(.degraded(operation: "start", detail: "\(error)"))
        }
    }

    /// The system-prompt fragment for a session.
    public func instructions(for context: MemorySessionContext) -> String {
        MemoryPrompt.instructions(scope: context.scope,
                                  session: context.session,
                                  bootstrap: context.bootstrap,
                                  isDurable: context.isDurable,
                                  tools: toolDefinitions().map(\.name))
    }

    /// The tool definitions to advertise, or none when tools are off.
    public func toolDefinitions() -> [MemoryToolDefinition] {
        guard configuration.isEnabled else { return [] }
        return MemoryTools.definitions(surface: configuration.toolSurface)
    }

    /// Runs one memory tool call in a session's scope.
    ///
    /// The scope comes from the session context, never from the call, so a
    /// model cannot reach another workspace by naming one.
    public func execute(name: String,
                        arguments: [String: MemoryToolValue],
                        in context: MemorySessionContext) async -> MemoryToolResult {
        guard configuration.isEnabled else { return .failure("memory is disabled") }
        let store = activeStore()
        let result = await MemoryTools.execute(name: name,
                                               arguments: arguments,
                                               store: store,
                                               scope: context.scope,
                                               session: context.session,
                                               limits: configuration.limits)
        if case .failure(let message) = result {
            log(.toolFailed(tool: name, detail: message))
            // A durable backend that failed sends later work to the local
            // store, and marks the session as no longer persisting.
            if durableStore != nil, !isDegraded, message.contains("unavailable")
                || message.contains("timed out") {
                isDegraded = true
                log(.degraded(operation: name, detail: message))
                if configuration.degradesToLocalStore {
                    return await MemoryTools.execute(name: name,
                                                     arguments: arguments,
                                                     store: localStore,
                                                     scope: context.scope,
                                                     session: context.session,
                                                     limits: configuration.limits)
                }
            }
        } else {
            log(.toolSucceeded(tool: name))
        }
        return result
    }

    /// Ends a session. With consolidation off this only logs; the hook for
    /// asking the model what to keep lives in the engine, which owns
    /// generation.
    public func endSession(_ context: MemorySessionContext) async {
        log(.sessionEnded(session: context.session.id, scope: context.scope))
    }

    /// Stores a consolidation the engine produced at session end.
    public func storeConsolidation(_ records: [MemoryRecord],
                                   in context: MemorySessionContext) async -> Int {
        let store = activeStore()
        var written = 0
        for record in records {
            var stamped = record
            stamped.sourceSession = context.session.id
            do {
                try await store.set(stamped, in: context.scope)
                written += 1
            } catch {
                log(.toolFailed(tool: "consolidation", detail: "\(error)"))
            }
        }
        log(.consolidated(session: context.session.id, records: written))
        return written
    }

    private func activeStore() -> any MemoryStore {
        guard let durableStore, !isDegraded else { return localStore }
        return durableStore
    }
}

/// Everything a session needs to carry once memory has started.
public struct MemorySessionContext: Sendable, Equatable {
    public let session: MemorySession
    public let scope: MemoryScope
    public let bootstrap: MemoryBootstrap
    /// False when the session is running on the local fallback, which the
    /// prompt tells the model so it does not promise persistence.
    public let isDurable: Bool

    public init(session: MemorySession,
                scope: MemoryScope,
                bootstrap: MemoryBootstrap,
                isDurable: Bool) {
        self.session = session
        self.scope = scope
        self.bootstrap = bootstrap
        self.isDurable = isDurable
    }
}

/// Observable memory events. The engine maps these onto its own log; keeping
/// them as values means this module prints nothing itself and stays testable.
public enum MemoryLogEvent: Sendable, Equatable {
    case sessionStarted(session: String, scope: MemoryScope, bootstrapRecords: Int,
                        bootstrapBytes: Int)
    case sessionEnded(session: String, scope: MemoryScope)
    case toolSucceeded(tool: String)
    case toolFailed(tool: String, detail: String)
    case degraded(operation: String, detail: String)
    case rejectedScope(String)
    case consolidated(session: String, records: Int)
    case journaled(session: String, index: Int, bytes: Int)

    /// One log line. Never contains a memory's contents or a credential: the
    /// log is operational, and memory can hold anything the model wrote.
    public var message: String {
        switch self {
        case .sessionStarted(let session, let scope, let records, let bytes):
            return "memory session=\(session) scope=\(scope.namespace)/\(scope.user)/"
                + "\(scope.workspace) bootstrap=\(records) records \(bytes)B"
        case .sessionEnded(let session, _):
            return "memory session=\(session) ended"
        case .toolSucceeded(let tool):
            return "memory tool=\(tool) ok"
        case .toolFailed(let tool, let detail):
            return "memory tool=\(tool) failed: \(detail)"
        case .degraded(let operation, let detail):
            return "memory degraded during \(operation): \(detail)"
        case .rejectedScope(let workspace):
            return "memory disabled for this session: unusable workspace '\(workspace)'"
        case .consolidated(let session, let records):
            return "memory session=\(session) consolidated \(records) records"
        case .journaled(let session, let index, let bytes):
            return "journal session=\(session) turn=\(index) \(bytes)B"
        }
    }
}
