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
    /// A store supplied by the caller, used for every scope. Nil in normal
    /// operation; this is how the tests drive the service.
    private let injectedStore: (any MemoryStore)?
    private let injectedJournal: (any SessionJournal)?
    /// One engine, one journal file and one workspace lock per scope.
    ///
    /// Not one engine for the whole service: a request that names another
    /// workspace would otherwise have its facts written into the default
    /// workspace's file, so deleting one project's memory would delete
    /// another's, and the "one file per workspace" the documentation promises
    /// would be false.
    private var workspaces: [MemoryScope: Workspace] = [:]
    /// When each workspace was last used, for deciding which to let go of.
    private var lastUsed: [MemoryScope: Date] = [:]
    private let localStore: InMemoryStore
    /// The engine-authored journal. Separate store, separate key space,
    /// separate trim policy: a busy week of sessions must never evict the
    /// facts the model wrote deliberately.
    private let journalFilter: JournalFilter
    /// Set once a durable operation has failed, so the session prompt can say
    /// memory is not persisting instead of the model assuming it is.
    private var isDegraded = false
    private var log: @Sendable (MemoryLogEvent) -> Void

    /// Everything one scope needs, created on first use.
    private struct Workspace {
        let store: any MemoryStore
        let journal: (any SessionJournal)?
        /// Nil when the caller injected its own store, in which case this
        /// service owns no engine to close.
        let engine: ContinuityEngine?
        /// False when the engine has no journal behind it, so writes last
        /// only as long as the process.
        let persists: Bool
    }

    public init(configuration: MemoryConfiguration,
                durableStore: (any MemoryStore)? = nil,
                journal: (any SessionJournal)? = nil,
                log: @escaping @Sendable (MemoryLogEvent) -> Void = { _ in }) {
        self.configuration = configuration
        self.localStore = InMemoryStore(limits: configuration.limits)
        self.journalFilter = configuration.journalLimits.filter
        self.log = log
        self.injectedStore = durableStore
        self.injectedJournal = journal
    }

    /// The engine, store and journal for a scope, built on first use.
    ///
    /// Deferred rather than done in `init` because opening a journal is I/O
    /// that can fail, and an initializer cannot report that to the caller who
    /// will actually be affected by it.
    private func workspace(for scope: MemoryScope) async -> Workspace? {
        if let existing = workspaces[scope] {
            lastUsed[scope] = Date()
            return existing
        }
        guard configuration.isEnabled else { return nil }

        if let injectedStore {
            let workspace = Workspace(store: injectedStore, journal: injectedJournal,
                                      engine: nil, persists: true)
            workspaces[scope] = workspace
            return workspace
        }

        let (engine, persists) = Self.makeEngine(configuration: configuration,
                                                 scope: scope, log: log)
        do {
            try await engine.start()
        } catch {
            isDegraded = true
            log(.degraded(operation: "start", detail: "\(error)"))
        }
        let store = ContinuityStore(engine: engine, limits: configuration.limits)
        var journal: (any SessionJournal)?
        if let injectedJournal {
            journal = injectedJournal
        } else if configuration.journalEnabled {
            journal = ContinuityJournalStore(engine: engine, store: store,
                                             limits: configuration.journalLimits)
        }
        let workspace = Workspace(store: store, journal: journal, engine: engine,
                                  persists: persists)
        workspaces[scope] = workspace
        lastUsed[scope] = Date()
        await enforceResidencyBudget(keeping: scope)
        return workspace
    }

    /// Keeps the whole subsystem inside the ceiling the machine was sized for.
    ///
    /// The ceiling is what memory adds to the process, not what each workspace
    /// may take. Per-workspace limits alone would multiply it by the number of
    /// workspaces a session has touched, so on an 8 GB machine "the model's
    /// budget plus 256 MiB" would quietly become plus 256 MiB per repository.
    ///
    /// Over the ceiling, the least recently used workspace is closed. Nothing
    /// is lost: everything it held is in its journal, and touching that
    /// workspace again replays it. The workspace in use is never closed.
    private func enforceResidencyBudget(keeping scope: MemoryScope) async {
        guard let ceiling = configuration.storage.maximumMemoryBytes, ceiling > 0 else { return }
        while workspaces.count > 1, await residentBytes() > ceiling {
            let candidates = lastUsed
                .filter { $0.key != scope && workspaces[$0.key] != nil }
                .sorted { $0.value < $1.value }
            guard let oldest = candidates.first?.key else { return }
            await workspaces[oldest]?.engine?.shutDown()
            workspaces[oldest] = nil
            lastUsed[oldest] = nil
            log(.degraded(operation: "residency",
                          detail: "closed workspace \(oldest.workspace) to stay inside "
                              + "\(ceiling >> 20) MiB"))
        }
    }

    /// Bytes memory is holding in this process, across every open workspace.
    public func residentBytes() async -> Int {
        var total = 0
        for workspace in workspaces.values {
            total += await workspace.engine?.residentBytes() ?? 0
        }
        return total
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
        scope: MemoryScope,
        log: @Sendable (MemoryLogEvent) -> Void
    ) -> (engine: ContinuityEngine, persists: Bool) {
        let budget = configuration.storage.budget
        let limits = ContinuityCore.MemoryLimits(
            maxValueBytes: configuration.limits.maximumValueBytes,
            maxBytesPerTask: budget.factBytes)
        let engineConfiguration = ContinuityConfiguration(
            memoryLimits: limits,
            sessionLogOptions: SessionLogOptions(maxBytesPerTask: budget.logBytes),
            journalsSessionContent: configuration.journalEnabled)
        do {
            let journal = try FileJournal(
                url: configuration.storage.journalURL(for: scope),
                synchronizesEveryWrite: configuration.storage.synchronizesEveryWrite)
            return (ContinuityEngine(configuration: engineConfiguration, journal: journal), true)
        } catch {
            // A journal held by another server is the expected case here, not
            // a broken install. Either way the session runs without
            // persistence and says so rather than writing into a file someone
            // else is also writing.
            log(.degraded(operation: "openJournal", detail: "\(error)"))
            return (ContinuityEngine(configuration: engineConfiguration), false)
        }
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
        guard let journal = await workspace(for: session.scope)?.journal else { return }
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
        await enforceResidencyBudget(keeping: session.scope)
    }

    /// Open the configured workspace now, so its journal is replayed at boot
    /// rather than on the first request.
    ///
    /// Replay is the one bulk read the store ever does. Paying it at start,
    /// while nothing is being generated, keeps it off the same disk the
    /// expert streamer is about to saturate and off the first user's
    /// latency. Safe to call more than once and safe with memory disabled.
    public func warmUp() async {
        guard let scope = configuration.scope() else { return }
        _ = await workspace(for: scope)
    }

    /// Close every workspace, flushing and releasing the workspace locks.
    ///
    /// A workspace's journal holds an exclusive lock for as long as it is
    /// open, so a process that is finished with a workspace has to say so.
    /// Leaving it to deallocation would make the moment another server can
    /// take over depend on when ARC happens to release an actor.
    public func shutDown() async {
        for workspace in workspaces.values {
            await workspace.engine?.shutDown()
        }
        workspaces.removeAll()
        lastUsed.removeAll()
    }

    /// The journal, for a caller that wants to read it back. Never used to
    /// build a prompt.
    public func journalStore(for scope: MemoryScope? = nil) async -> (any SessionJournal)? {
        guard let resolved = scope ?? configuration.scope() else { return nil }
        return await workspace(for: resolved)?.journal
    }

    public var isEnabled: Bool { configuration.isEnabled }

    /// Whether writes reach durable storage in a scope. False once a durable
    /// operation has failed, and false when the journal could not be opened
    /// at all.
    public func isDurable(in scope: MemoryScope) async -> Bool {
        guard !isDegraded else { return false }
        return await workspace(for: scope)?.persists ?? false
    }

    /// Whether the configuration's own scope is persisting.
    public var isDurable: Bool {
        get async {
            guard let scope = configuration.scope() else { return false }
            return await isDurable(in: scope)
        }
    }

    /// Starts a session and returns what the engine needs to install.
    ///
    /// A failure here degrades rather than propagates: the session continues
    /// with local memory when that is allowed, and with none when it is not.
    /// - Parameter tag: what the session is about, when the caller could
    ///   tell. Recorded on the session, shown in the log; not a scope.
    public func beginSession(id: String,
                             workspaceOverride: String? = nil,
                             modelID: String? = nil,
                             tag: String? = nil) async -> MemorySessionContext? {
        guard configuration.isEnabled else { return nil }
        guard let scope = configuration.scope(workspaceOverride: workspaceOverride) else {
            log(.rejectedScope(workspaceOverride ?? configuration.workspace))
            return nil
        }
        let session = MemorySession(id: id, modelID: modelID, tag: tag)
        var bootstrap = MemoryBootstrap.empty
        if let workspace = await workspace(for: scope) {
            do {
                bootstrap = try await workspace.store.sessionInit(session, in: scope)
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
        let durable = await isDurable(in: scope)
        log(.sessionStarted(session: session.id, scope: scope,
                            bootstrapRecords: bootstrap.records.count,
                            bootstrapBytes: bootstrap.totalBytes))
        return MemorySessionContext(session: session,
                                    scope: scope,
                                    bootstrap: bootstrap,
                                    isDurable: durable)
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
        let store = await activeStore(for: context.scope)
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
            if !isDegraded, message.contains("unavailable")
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
            // Checked after the write, not only when a workspace is opened.
            // A ceiling that only holds while the set of workspaces is
            // changing is not a ceiling.
            await enforceResidencyBudget(keeping: context.scope)
        }
        return result
    }

    /// Ends a session. With consolidation off this only logs; the hook for
    /// asking the model what to keep lives in the engine, which owns
    /// generation.
    public func endSession(_ context: MemorySessionContext) async {
        log(.sessionEnded(session: context.session.id, scope: context.scope))
    }

    /// Facts already in a scope, most important first, so a consolidation
    /// can update an address instead of inventing a near-duplicate beside
    /// it -- and can see the value it would be replacing.
    ///
    /// Values, not only keys. Shown keys alone, a model re-derived every one
    /// of them from a session that said nothing about them, and wrote "not
    /// specified" over a character's eye colour.
    public func recordedFacts(in scope: MemoryScope, limit: Int = 60) async -> [MemoryRecord] {
        let store = await activeStore(for: scope)
        return (try? await store.search(MemoryQuery(limit: limit), in: scope)) ?? []
    }

    /// Stores a consolidation the engine produced at session end.
    public func storeConsolidation(_ records: [MemoryRecord],
                                   in context: MemorySessionContext) async -> Int {
        let store = await activeStore(for: context.scope)
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

    private func activeStore(for scope: MemoryScope) async -> any MemoryStore {
        guard !isDegraded, let workspace = await workspace(for: scope) else { return localStore }
        return workspace.store
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
