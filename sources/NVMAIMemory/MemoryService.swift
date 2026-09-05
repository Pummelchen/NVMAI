import Foundation

/// What the serving engine talks to.
///
/// It owns the choice of backend, the fallback when Valkey is unreachable,
/// and the session lifecycle. The engine calls four things: `beginSession`,
/// `instructions`, `execute` and `endSession`. Everything else stays here.
///
/// The service never throws at the engine. Memory is optional by design, so
/// a failure produces a logged event and a degraded mode, not a failed
/// completion. The one thing it will not do is report a write as successful
/// when it was not.
public actor MemoryService {
    public private(set) var configuration: MemoryConfiguration
    private let durableStore: (any MemoryStore)?
    private let localStore: InMemoryStore
    /// The engine-authored journal. Separate store, separate key space,
    /// separate trim policy: a busy week of sessions must never evict the
    /// facts the model wrote deliberately.
    private let journal: (any SessionJournal)?
    private let journalFilter: JournalFilter
    /// Set once a durable operation has failed, so the session prompt can say
    /// memory is not persisting instead of the model assuming it is.
    private var isDegraded = false
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
        } else if configuration.isEnabled {
            let connection = ValkeyConnection(configuration: configuration.valkey)
            self.durableStore = ValkeyMemoryStore(connection: connection,
                                                  prefix: "nvmai:mem",
                                                  limits: configuration.limits,
                                                  maximumIndexScan: configuration.maximumIndexScan)
        } else {
            self.durableStore = nil
        }
        if let journal {
            self.journal = journal
        } else if configuration.isEnabled, configuration.journalEnabled {
            // Its own connection: journal writes happen after a completion has
            // already been returned, and must never queue behind a read the
            // model is waiting on.
            self.journal = ValkeyJournal(connection: ValkeyConnection(
                configuration: configuration.valkey),
                                         prefix: "nvmai:mem",
                                         limits: configuration.journalLimits)
        } else {
            self.journal = nil
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
    public var isDurable: Bool { durableStore != nil && !isDegraded }

    /// Starts a session and returns what the engine needs to install.
    ///
    /// A failure here degrades rather than propagates: the session continues
    /// with local memory when that is allowed, and with none when it is not.
    public func beginSession(id: String,
                             workspaceOverride: String? = nil,
                             modelID: String? = nil) async -> MemorySessionContext? {
        guard configuration.isEnabled else { return nil }
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

    /// The system-prompt fragment for a session.
    public func instructions(for context: MemorySessionContext) -> String {
        MemoryPrompt.instructions(scope: context.scope,
                                  session: context.session,
                                  bootstrap: context.bootstrap,
                                  isDurable: context.isDurable)
    }

    /// The tool definitions to advertise, or none when tools are off.
    public func toolDefinitions() -> [MemoryToolDefinition] {
        guard configuration.isEnabled, configuration.exposesTools else { return [] }
        return MemoryTools.definitions()
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
