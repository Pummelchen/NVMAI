import Foundation

/// How the engine behaves.
public struct ContinuityConfiguration: Sendable {
    public var memoryLimits: MemoryLimits
    public var sessionLogOptions: SessionLogOptions
    public var defaultBudget: ContextBudget
    /// Write prompts and replies to the journal.
    ///
    /// Separate from journalling memory because the two differ in
    /// sensitivity: memory holds distilled facts, the session log holds
    /// everything a person typed. A caller who wants durable continuity
    /// without a transcript on disk turns this off and keeps the rest.
    public var journalsSessionContent: Bool
    /// Compact the journal once it holds more than this many records.
    /// Zero disables automatic compaction.
    public var compactionThreshold: Int

    public init(memoryLimits: MemoryLimits = .default,
                sessionLogOptions: SessionLogOptions = .init(),
                defaultBudget: ContextBudget = ContextBudget(),
                journalsSessionContent: Bool = true,
                compactionThreshold: Int = 20_000) {
        self.memoryLimits = memoryLimits
        self.sessionLogOptions = sessionLogOptions
        self.defaultBudget = defaultBudget
        self.journalsSessionContent = journalsSessionContent
        self.compactionThreshold = compactionThreshold
    }
}

/// The public face of the package.
///
/// It owns a `SessionLog`, a `TaskMemory` and a `ContextAssembler`, wires the
/// journal to both stores, and offers the small set of calls an integration
/// actually needs. Everything underneath stays public, so an application that
/// outgrows this façade can drive the pieces directly rather than fork it.
///
/// Nothing here opens a socket or spawns a process. The engine runs in the
/// caller's process, and the only thing it touches outside memory is the
/// journal file it was handed.
public actor ContinuityEngine {
    public let sessionLog: SessionLog
    public let memory: TaskMemory
    private let journal: ContinuityJournal
    private var assembler: ContextAssembler
    private let configuration: ContinuityConfiguration
    private var journaledRecords = 0
    private var started = false

    public init(configuration: ContinuityConfiguration = ContinuityConfiguration(),
                journal: ContinuityJournal = NullJournal(),
                assembler: ContextAssembler = DefaultContextAssembler()) {
        self.configuration = configuration
        self.journal = journal
        self.assembler = assembler
        self.sessionLog = SessionLog(options: configuration.sessionLogOptions)
        self.memory = TaskMemory(limits: configuration.memoryLimits)
    }

    /// Replay the journal and begin recording.
    ///
    /// Must be called before anything else when a journal is in use.
    /// Recording is wired only after the replay, so restored state is not
    /// written back to the file it came from.
    public func start() async throws {
        guard !started else { return }
        started = true
        try await restoreFromJournal()
        await installObservers()
    }

    // MARK: - Tasks and sessions

    @discardableResult
    public func createTask(title: String,
                           objective: String = "",
                           id: UUID = UUID()) async throws -> ContinuityTask {
        let task = await sessionLog.createTask(title: title, objective: objective, id: id)
        try await record(.task(task))
        return task
    }

    public func task(_ id: UUID) async -> ContinuityTask? {
        await sessionLog.task(id)
    }

    public func tasks() async -> [ContinuityTask] {
        await sessionLog.tasks()
    }

    @discardableResult
    public func updateTask(_ id: UUID,
                           title: String? = nil,
                           objective: String? = nil) async throws -> ContinuityTask {
        let task = try await sessionLog.updateTask(id, title: title, objective: objective)
        try await record(.task(task))
        return task
    }

    @discardableResult
    public func beginSession(taskID: UUID,
                             model: String? = nil,
                             externalID: String? = nil) async throws -> Session {
        let session = try await sessionLog.beginSession(taskID: taskID, model: model,
                                                        externalID: externalID)
        try await record(.session(session))
        return session
    }

    @discardableResult
    public func endSession(_ id: UUID) async throws -> Session {
        let session = try await sessionLog.endSession(id)
        try await record(.session(session))
        // A session boundary is the natural durability point: it is the
        // moment where losing the last few records would actually cost
        // something, and it is rare enough that the barrier is free.
        try? await flush()
        return session
    }

    /// Force everything written so far to disk.
    public func flush() async throws {
        guard let journal = journal as? FileJournal else { return }
        try await journal.sync()
    }

    /// Flush, close the journal and release the workspace lock.
    ///
    /// A `FileJournal` holds an exclusive lock on its workspace for as long
    /// as it exists, so a caller that wants to hand the workspace to another
    /// engine — or to another process — has to say when it is finished.
    /// Waiting for deallocation is not a contract anyone can rely on.
    ///
    /// The engine keeps working afterwards; it simply stops persisting.
    public func shutDown() async {
        try? await flush()
        guard let journal = journal as? FileJournal else { return }
        try? await journal.shutDown()
    }

    // MARK: - Recording

    public func recordUserPrompt(sessionID: UUID, text: String) async throws {
        _ = try await sessionLog.recordUserPrompt(sessionID: sessionID, text: text)
    }

    public func recordAssistantResponse(sessionID: UUID,
                                        text: String,
                                        model: String? = nil,
                                        inputTokens: Int? = nil,
                                        outputTokens: Int? = nil,
                                        latencyMilliseconds: Int? = nil,
                                        finishReason: String? = nil) async throws {
        let record = ResponseRecord(text: text,
                                    model: model,
                                    inputTokens: inputTokens,
                                    outputTokens: outputTokens,
                                    latencyMilliseconds: latencyMilliseconds,
                                    finishReason: finishReason)
        _ = try await sessionLog.recordAssistantResponse(sessionID: sessionID, record)
    }

    @discardableResult
    public func beginAssistantResponse(sessionID: UUID,
                                       model: String? = nil,
                                       requestID: String? = nil) async throws -> UUID {
        try await sessionLog.beginAssistantResponse(sessionID: sessionID,
                                                    model: model,
                                                    requestID: requestID)
    }

    public func appendAssistantChunk(responseID: UUID, text: String) async throws {
        try await sessionLog.appendAssistantChunk(responseID: responseID, text: text)
    }

    public func completeAssistantResponse(responseID: UUID,
                                          inputTokens: Int? = nil,
                                          outputTokens: Int? = nil,
                                          finishReason: String? = nil) async throws {
        _ = try await sessionLog.completeAssistantResponse(responseID: responseID,
                                                           inputTokens: inputTokens,
                                                           outputTokens: outputTokens,
                                                           finishReason: finishReason)
    }

    // MARK: - Memory

    /// Write a fact, attributed to the session that produced it.
    ///
    /// The session is what makes the write explainable later, so it is not
    /// optional here even though `TaskMemory` allows it: a fact with no
    /// origin is exactly the kind of state that becomes unarguable in month
    /// three of a long task.
    @discardableResult
    public func remember(sessionID: UUID,
                         namespace: String,
                         key: String,
                         value: String,
                         author: ProvenanceAuthor = .model,
                         eventID: UUID? = nil,
                         importance: Double? = nil,
                         confidence: Double? = nil,
                         tags: [String]? = nil,
                         dependencies: [String]? = nil,
                         expectedVersion: Int? = nil) async throws -> MemoryWriteResult {
        guard let session = await sessionLog.session(sessionID) else {
            throw ContinuityError.unknownSession(sessionID)
        }
        let provenance = Provenance(sessionID: sessionID, eventID: eventID, author: author)
        let result = try await memory.write(taskID: session.taskID,
                                            namespace: namespace,
                                            key: key,
                                            value: value,
                                            provenance: provenance,
                                            importance: importance,
                                            confidence: confidence,
                                            tags: tags,
                                            dependencies: dependencies,
                                            expectedVersion: expectedVersion)
        _ = try? await sessionLog.recordMemoryWrite(sessionID: sessionID, item: result.item)
        return result
    }

    /// Write a fact that no session produced, such as a task's opening state.
    @discardableResult
    public func remember(taskID: UUID,
                         namespace: String,
                         key: String,
                         value: String,
                         author: ProvenanceAuthor = .engine,
                         importance: Double? = nil,
                         confidence: Double? = nil,
                         tags: [String]? = nil,
                         dependencies: [String]? = nil) async throws -> MemoryWriteResult {
        guard await sessionLog.task(taskID) != nil else {
            throw ContinuityError.unknownTask(taskID)
        }
        return try await memory.write(taskID: taskID,
                                      namespace: namespace,
                                      key: key,
                                      value: value,
                                      provenance: Provenance(author: author),
                                      importance: importance,
                                      confidence: confidence,
                                      tags: tags,
                                      dependencies: dependencies)
    }

    public func recall(taskID: UUID, _ query: MemoryQuery = .active) async -> [MemoryItem] {
        await memory.query(taskID: taskID, query)
    }

    public func recall(taskID: UUID, namespace: String, key: String) async -> MemoryItem? {
        await memory.item(taskID: taskID, namespace: namespace, key: key)
    }

    public func history(taskID: UUID, namespace: String, key: String) async -> [MemoryVersion] {
        await memory.history(taskID: taskID, namespace: namespace, key: key)
    }

    /// Retire a fact without destroying it.
    @discardableResult
    public func archive(taskID: UUID, namespace: String, key: String) async throws -> MemoryItem {
        try await memory.archive(taskID: taskID, namespace: namespace, key: key)
    }

    /// Mark a fact as contradicted. It keeps its value and is still offered
    /// to the model, flagged, because the model is the thing best placed to
    /// resolve it and cannot do so if the conflict is hidden.
    @discardableResult
    public func dispute(taskID: UUID, namespace: String, key: String) async throws -> MemoryItem {
        try await memory.setStatus(taskID: taskID, namespace: namespace, key: key,
                                   status: .disputed)
    }

    @discardableResult
    public func resolve(taskID: UUID, namespace: String, key: String) async throws -> MemoryItem {
        try await memory.setStatus(taskID: taskID, namespace: namespace, key: key,
                                   status: .active)
    }

    // MARK: - Context

    public func setAssembler(_ assembler: ContextAssembler) {
        self.assembler = assembler
    }

    /// Build the block of text to put in front of the model.
    ///
    /// Records a `contextAssembled` event when a session is given, so the log
    /// can later show what the model was looking at when it answered.
    public func assembleContext(taskID: UUID,
                                sessionID: UUID? = nil,
                                focus: String? = nil,
                                budget: ContextBudget? = nil,
                                query: MemoryQuery = .active) async throws -> ContextSnapshot {
        guard let task = await sessionLog.task(taskID) else {
            throw ContinuityError.unknownTask(taskID)
        }
        let effectiveBudget = budget ?? configuration.defaultBudget
        let candidates = await memory.query(taskID: taskID, query)
        let all = await memory.query(taskID: taskID,
                                     MemoryQuery(statuses: [.active, .disputed],
                                                 order: .address))
        var index: [String: MemoryItem] = [:]
        for item in all { index[item.address] = item }
        let turns = effectiveBudget.recentTurnCount > 0
            ? await sessionLog.turns(taskID: taskID, limit: effectiveBudget.recentTurnCount)
            : []
        let request = ContextRequest(task: task,
                                     sessionID: sessionID,
                                     items: candidates,
                                     index: index,
                                     turns: turns,
                                     budget: effectiveBudget,
                                     focus: focus)
        let snapshot = try assembler.assemble(request)
        if let sessionID {
            _ = try? await sessionLog.recordContextAssembled(sessionID: sessionID,
                                                             snapshot: snapshot)
        }
        return snapshot
    }

    // MARK: - Reading the log

    public func events(sessionID: UUID) async -> [SessionEvent] {
        await sessionLog.events(sessionID: sessionID)
    }

    public func turns(taskID: UUID, limit: Int? = nil) async -> [SessionTurn] {
        await sessionLog.turns(taskID: taskID, limit: limit)
    }

    public func sessions(taskID: UUID) async -> [Session] {
        await sessionLog.sessions(taskID: taskID)
    }

    public func session(externalID: String, taskID: UUID? = nil) async -> Session? {
        await sessionLog.session(externalID: externalID, taskID: taskID)
    }

    /// Retention for a task's sessions. See `SessionLog.pruneSessions`.
    @discardableResult
    public func pruneSessions(taskID: UUID, keeping: Int) async -> [UUID] {
        await sessionLog.pruneSessions(taskID: taskID, keeping: keeping)
    }

    /// Retention for one session's turns. See `SessionLog.pruneTurns`.
    @discardableResult
    public func pruneTurns(sessionID: UUID, keeping: Int) async -> Int {
        await sessionLog.pruneTurns(sessionID: sessionID, keeping: keeping)
    }

    // MARK: - Maintenance

    /// Bytes this engine currently holds, across every task.
    ///
    /// Cheap on purpose: a caller policing a process-wide budget has to be
    /// able to ask often, and `statistics()` walks every event to count them.
    public func residentBytes() async -> Int {
        var total = 0
        for taskID in await sessionLog.taskIdentifiers() {
            total += await memory.byteCount(taskID: taskID)
            total += await sessionLog.byteCount(taskID: taskID)
        }
        return total
    }

    public func statistics() async -> ContinuityStatistics {
        let tasks = await sessionLog.tasks()
        var sessionCount = 0
        var eventCount = 0
        var itemCount = 0
        var memoryBytes = 0
        var logBytes = 0
        for task in tasks {
            let sessions = await sessionLog.sessions(taskID: task.id)
            sessionCount += sessions.count
            eventCount += await sessionLog.events(taskID: task.id).count
            itemCount += await memory.count(taskID: task.id)
            memoryBytes += await memory.byteCount(taskID: task.id)
            logBytes += await sessionLog.byteCount(taskID: task.id)
        }
        return ContinuityStatistics(taskCount: tasks.count,
                                    sessionCount: sessionCount,
                                    eventCount: eventCount,
                                    memoryItemCount: itemCount,
                                    memoryBytes: memoryBytes,
                                    logBytes: logBytes,
                                    journaledRecords: journaledRecords)
    }

    /// Collapse the journal to a single checkpoint of current state.
    public func compactJournal() async throws {
        let log = await sessionLog.snapshot()
        let store = await memory.snapshot()
        try await journal.compact(sessionLog: log, memory: store)
        journaledRecords = 1
    }

    /// Delete a task, its sessions, its log and its memory.
    ///
    /// The journal is compacted immediately afterwards, because a delete that
    /// leaves the content sitting in an append-only file is not a delete.
    public func forget(taskID: UUID) async throws {
        await sessionLog.forget(taskID: taskID)
        await memory.forget(taskID: taskID)
        try await compactJournal()
    }

    // MARK: - Internals

    private func installObservers() async {
        let journal = self.journal
        let journalsContent = configuration.journalsSessionContent

        await sessionLog.setObserver { [weak self] event in
            guard journalsContent || !Self.carriesContent(event) else { return }
            try? await journal.append(.event(event))
            await self?.countRecord()
        }
        await memory.setObserver { [weak self] mutation in
            switch mutation {
            case .versioned(let version):
                try? await journal.append(.memoryVersion(version))
            case .written(let result):
                try? await journal.append(.memory(result.item))
            case .statusChanged(let item):
                try? await journal.append(.memory(item))
            }
            await self?.countRecord()
        }
    }

    /// Whether an event carries what a person or the model actually wrote, as
    /// opposed to structure and measurements.
    private static func carriesContent(_ event: SessionEvent) -> Bool {
        switch event.kind {
        case .userPrompt, .assistantResponse, .assistantResponseChunk,
             .assistantResponseCompleted:
            return true
        case .sessionStarted, .sessionEnded, .assistantResponseStarted,
             .memoryWritten, .contextAssembled:
            return false
        }
    }

    private func countRecord() async {
        journaledRecords += 1
        guard configuration.compactionThreshold > 0,
              journaledRecords > configuration.compactionThreshold else { return }
        try? await compactJournal()
    }

    private func record(_ entry: JournalRecord) async throws {
        try await journal.append(entry)
        journaledRecords += 1
    }

    private func restoreFromJournal() async throws {
        let records = try await journal.replay()
        guard !records.isEmpty else { return }
        var log = SessionLogSnapshot()
        var store = MemorySnapshot()
        var itemsByAddress: [String: MemoryItem] = [:]
        var tasksByID: [UUID: ContinuityTask] = [:]
        var sessionsByID: [UUID: Session] = [:]

        for entry in records {
            switch entry {
            case .checkpoint(let logSnapshot, let memorySnapshot):
                log = logSnapshot
                store = memorySnapshot
                tasksByID = Dictionary(uniqueKeysWithValues: logSnapshot.tasks.map { ($0.id, $0) })
                sessionsByID = Dictionary(uniqueKeysWithValues:
                                            logSnapshot.sessions.map { ($0.id, $0) })
                itemsByAddress = [:]
                for item in memorySnapshot.items {
                    itemsByAddress["\(item.taskID)/\(item.address)"] = item
                }
            case .task(let task):
                tasksByID[task.id] = task
            case .session(let session):
                sessionsByID[session.id] = session
            case .event(let event):
                log.events.append(event)
            case .memory(let item):
                itemsByAddress["\(item.taskID)/\(item.address)"] = item
            case .memoryVersion(let version):
                store.versions.append(version)
            }
        }
        log.tasks = Array(tasksByID.values)
        log.sessions = Array(sessionsByID.values)
        store.items = Array(itemsByAddress.values)
        await sessionLog.restore(log)
        await memory.restore(store)
        journaledRecords = records.count
    }
}

public struct ContinuityStatistics: Sendable, Equatable {
    public let taskCount: Int
    public let sessionCount: Int
    public let eventCount: Int
    public let memoryItemCount: Int
    /// Bytes of task memory held in this process.
    public let memoryBytes: Int
    /// Bytes of session log held in this process. The journal file may hold
    /// more; this is what is resident.
    public let logBytes: Int
    public let journaledRecords: Int
}
