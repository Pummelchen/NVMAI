import Foundation

/// The complete, append-only record of what happened.
///
/// The log costs the model nothing. Nothing here is written by a tool call,
/// so no tokens are spent recording it and no session can fail to record
/// itself by declining to cooperate. That is the whole reason the log and the
/// memory are separate: memory is what the model decides is worth keeping,
/// the log is what actually occurred.
///
/// Everything is engine-authored and nothing is mutated after the fact. The
/// only removal is `forget(taskID:)`, which exists so a user can delete a
/// project outright.
public actor SessionLog {
    private var tasks: [UUID: ContinuityTask] = [:]
    private var sessions: [UUID: Session] = [:]
    /// Events per session, in the order they were appended.
    private var events: [UUID: [SessionEvent]] = [:]
    /// Sessions per task, oldest first.
    private var sessionsByTask: [UUID: [UUID]] = [:]
    /// Streaming replies still open, keyed by response identifier.
    private var openResponses: [UUID: OpenResponse] = [:]
    /// Bytes held per task, kept incrementally.
    private var bytes: [UUID: Int] = [:]
    /// Bytes held per session, so pruning one can be subtracted exactly
    /// instead of triggering a full recount.
    private var sessionBytes: [UUID: Int] = [:]
    private let options: SessionLogOptions
    private var observer: (@Sendable (SessionEvent) async -> Void)?

    public init(options: SessionLogOptions = .init()) {
        self.options = options
    }

    /// Called after each appended event, so the engine can journal without
    /// this actor knowing what persistence is.
    public func setObserver(_ observer: (@Sendable (SessionEvent) async -> Void)?) {
        self.observer = observer
    }

    // MARK: - Tasks

    @discardableResult
    public func createTask(title: String,
                           objective: String = "",
                           id: UUID = UUID(),
                           now: Date = Date()) -> ContinuityTask {
        let task = ContinuityTask(id: id, title: title, objective: objective,
                                  createdAt: now, updatedAt: now)
        tasks[task.id] = task
        return task
    }

    public func task(_ id: UUID) -> ContinuityTask? { tasks[id] }

    public func tasks(_ ids: [UUID]? = nil) -> [ContinuityTask] {
        let all = Array(tasks.values).sorted { $0.createdAt < $1.createdAt }
        guard let ids else { return all }
        let wanted = Set(ids)
        return all.filter { wanted.contains($0.id) }
    }

    @discardableResult
    public func updateTask(_ id: UUID,
                           title: String? = nil,
                           objective: String? = nil,
                           now: Date = Date()) throws -> ContinuityTask {
        guard var task = tasks[id] else { throw ContinuityError.unknownTask(id) }
        if let title { task.title = title }
        if let objective { task.objective = objective }
        task.updatedAt = now
        tasks[id] = task
        return task
    }

    // MARK: - Sessions

    @discardableResult
    public func beginSession(taskID: UUID,
                             model: String? = nil,
                             externalID: String? = nil,
                             id: UUID = UUID(),
                             now: Date = Date()) async throws -> Session {
        guard tasks[taskID] != nil else { throw ContinuityError.unknownTask(taskID) }
        let session = Session(id: id, taskID: taskID, startedAt: now, model: model,
                              externalID: externalID)
        sessions[session.id] = session
        sessionsByTask[taskID, default: []].append(session.id)
        events[session.id] = []
        await append(SessionEvent(sessionID: session.id, taskID: taskID, timestamp: now,
                                  kind: .sessionStarted,
                                  payload: model.map { .text($0) } ?? .none))
        return session
    }

    @discardableResult
    public func endSession(_ id: UUID, now: Date = Date()) async throws -> Session {
        guard var session = sessions[id] else { throw ContinuityError.unknownSession(id) }
        guard session.isOpen else { throw ContinuityError.sessionAlreadyEnded(id) }
        // A reply still streaming when the session ends is closed with what
        // arrived. Dropping it would lose the only copy.
        for (responseID, open) in openResponses where open.sessionID == id {
            _ = try? await completeAssistantResponse(responseID: responseID, now: now)
        }
        session.endedAt = now
        sessions[id] = session
        await append(SessionEvent(sessionID: id, taskID: session.taskID, timestamp: now,
                                  kind: .sessionEnded, payload: .none))
        return session
    }

    public func session(_ id: UUID) -> Session? { sessions[id] }

    public func sessions(taskID: UUID) -> [Session] {
        (sessionsByTask[taskID] ?? []).compactMap { sessions[$0] }
    }

    /// The session a caller named, if it is still known.
    public func session(externalID: String, taskID: UUID? = nil) -> Session? {
        sessions.values.first {
            $0.externalID == externalID && (taskID == nil || $0.taskID == taskID)
        }
    }

    // MARK: - Recording

    @discardableResult
    public func recordUserPrompt(sessionID: UUID,
                                 text: String,
                                 now: Date = Date()) async throws -> SessionEvent {
        let session = try requireOpenSession(sessionID)
        let event = SessionEvent(sessionID: sessionID, taskID: session.taskID,
                                 timestamp: now, kind: .userPrompt, payload: .text(text))
        await append(event)
        return event
    }

    /// Record a reply that is already complete.
    @discardableResult
    public func recordAssistantResponse(sessionID: UUID,
                                        _ record: ResponseRecord,
                                        now: Date = Date()) async throws -> SessionEvent {
        let session = try requireOpenSession(sessionID)
        let event = SessionEvent(sessionID: sessionID, taskID: session.taskID,
                                 timestamp: now, kind: .assistantResponse,
                                 payload: .response(record))
        await append(event)
        return event
    }

    // MARK: - Streaming replies

    /// Open a streamed reply and return the identifier its chunks belong to.
    @discardableResult
    public func beginAssistantResponse(sessionID: UUID,
                                       model: String? = nil,
                                       requestID: String? = nil,
                                       now: Date = Date()) async throws -> UUID {
        let session = try requireOpenSession(sessionID)
        let event = SessionEvent(sessionID: sessionID, taskID: session.taskID,
                                 timestamp: now, kind: .assistantResponseStarted,
                                 payload: .none)
        openResponses[event.id] = OpenResponse(sessionID: sessionID,
                                               taskID: session.taskID,
                                               model: model,
                                               requestID: requestID,
                                               startedAt: now)
        await append(event.withResponseID(event.id))
        return event.id
    }

    /// Add a piece of a streamed reply.
    ///
    /// The text is buffered, not appended as an event, so the completed reply
    /// appears exactly once in the log. A chunk event is written only when
    /// `SessionLogOptions.persistsChunks` is set, and readers then ignore
    /// chunks for any response that also has a completion.
    public func appendAssistantChunk(responseID: UUID,
                                     text: String,
                                     now: Date = Date()) async throws {
        guard var open = openResponses[responseID] else {
            throw ContinuityError.unknownSession(responseID)
        }
        open.buffer += text
        open.chunkCount += 1
        openResponses[responseID] = open
        guard options.persistsChunks else { return }
        await append(SessionEvent(sessionID: open.sessionID, taskID: open.taskID,
                                  timestamp: now, kind: .assistantResponseChunk,
                                  payload: .text(text), responseID: responseID))
    }

    /// Close a streamed reply, writing the assembled text as one event.
    @discardableResult
    public func completeAssistantResponse(responseID: UUID,
                                          inputTokens: Int? = nil,
                                          outputTokens: Int? = nil,
                                          finishReason: String? = nil,
                                          responseIdentifier: String? = nil,
                                          now: Date = Date()) async throws -> SessionEvent {
        guard let open = openResponses.removeValue(forKey: responseID) else {
            throw ContinuityError.unknownSession(responseID)
        }
        let latency = Int(now.timeIntervalSince(open.startedAt) * 1000)
        let record = ResponseRecord(text: open.buffer,
                                    model: open.model,
                                    requestID: open.requestID,
                                    responseID: responseIdentifier,
                                    inputTokens: inputTokens,
                                    outputTokens: outputTokens,
                                    latencyMilliseconds: latency,
                                    finishReason: finishReason)
        let event = SessionEvent(sessionID: open.sessionID, taskID: open.taskID,
                                 timestamp: now, kind: .assistantResponseCompleted,
                                 payload: .response(record), responseID: responseID)
        await append(event)
        return event
    }

    @discardableResult
    public func recordMemoryWrite(sessionID: UUID,
                                  item: MemoryItem,
                                  now: Date = Date()) async throws -> SessionEvent {
        let session = try requireSession(sessionID)
        let event = SessionEvent(sessionID: sessionID, taskID: session.taskID,
                                 timestamp: now, kind: .memoryWritten,
                                 payload: .memory(namespace: item.namespace,
                                                  key: item.key,
                                                  version: item.version,
                                                  itemID: item.id))
        await append(event)
        return event
    }

    @discardableResult
    public func recordContextAssembled(sessionID: UUID,
                                       snapshot: ContextSnapshot,
                                       now: Date = Date()) async throws -> SessionEvent {
        let session = try requireSession(sessionID)
        let event = SessionEvent(sessionID: sessionID, taskID: session.taskID,
                                 timestamp: now, kind: .contextAssembled,
                                 payload: .context(snapshotID: snapshot.id,
                                                   itemCount: snapshot.memoryItemIDs.count,
                                                   estimatedTokens: snapshot.estimatedTokenCount))
        await append(event)
        return event
    }

    // MARK: - Reading

    public func events(sessionID: UUID) -> [SessionEvent] { events[sessionID] ?? [] }

    /// Every event for a task, ordered by session then by append order.
    public func events(taskID: UUID) -> [SessionEvent] {
        (sessionsByTask[taskID] ?? []).flatMap { events[$0] ?? [] }
    }

    /// The session's content with streamed chunks folded away.
    public func transcript(sessionID: UUID) -> [SessionEvent] {
        Self.fold(events[sessionID] ?? [])
    }

    /// Prompt-and-reply pairs for a task, newest last.
    ///
    /// This is what a context assembler wants when it needs "what happened
    /// recently" rather than "what is true".
    public func turns(taskID: UUID, limit: Int? = nil) -> [SessionTurn] {
        var turns: [SessionTurn] = []
        var pendingPrompt: SessionEvent?
        for event in Self.fold(events(taskID: taskID)) {
            switch event.kind {
            case .userPrompt:
                if let prompt = pendingPrompt {
                    turns.append(SessionTurn(sessionID: prompt.sessionID,
                                             promptEventID: prompt.id,
                                             prompt: prompt.payload.text ?? "",
                                             response: nil,
                                             timestamp: prompt.timestamp))
                }
                pendingPrompt = event
            case .assistantResponse, .assistantResponseCompleted:
                var record: ResponseRecord?
                if case .response(let value) = event.payload { record = value }
                turns.append(SessionTurn(sessionID: event.sessionID,
                                         promptEventID: pendingPrompt?.id,
                                         prompt: pendingPrompt?.payload.text ?? "",
                                         response: event.payload.text,
                                         responseRecord: record,
                                         completedAt: event.timestamp,
                                         timestamp: pendingPrompt?.timestamp ?? event.timestamp))
                pendingPrompt = nil
            default:
                continue
            }
        }
        if let prompt = pendingPrompt {
            turns.append(SessionTurn(sessionID: prompt.sessionID,
                                     promptEventID: prompt.id,
                                     prompt: prompt.payload.text ?? "",
                                     response: nil,
                                     timestamp: prompt.timestamp))
        }
        if let limit, turns.count > limit {
            turns = Array(turns.suffix(limit))
        }
        return turns
    }

    /// Drops chunk events for any reply that also has a completion, so text
    /// that was streamed appears once.
    static func fold(_ input: [SessionEvent]) -> [SessionEvent] {
        let completed = Set(input.compactMap { event -> UUID? in
            event.kind == .assistantResponseCompleted ? event.responseID : nil
        })
        return input.filter { event in
            switch event.kind {
            case .assistantResponseChunk:
                guard let id = event.responseID else { return true }
                return !completed.contains(id)
            case .assistantResponseStarted:
                return false
            default:
                return true
            }
        }
    }

    // MARK: - Snapshot and restore

    public func snapshot() -> SessionLogSnapshot {
        SessionLogSnapshot(tasks: Array(tasks.values),
                           sessions: Array(sessions.values),
                           events: sessionsByTask.values.flatMap { ids in
                               ids.flatMap { events[$0] ?? [] }
                           })
    }

    public func restore(_ snapshot: SessionLogSnapshot) {
        tasks = Dictionary(uniqueKeysWithValues: snapshot.tasks.map { ($0.id, $0) })
        sessions = Dictionary(uniqueKeysWithValues: snapshot.sessions.map { ($0.id, $0) })
        events = [:]
        sessionsByTask = [:]
        openResponses = [:]
        for session in snapshot.sessions.sorted(by: { $0.startedAt < $1.startedAt }) {
            sessionsByTask[session.taskID, default: []].append(session.id)
            events[session.id] = []
        }
        for event in snapshot.events {
            events[event.sessionID, default: []].append(event)
        }
        bytes = [:]
        sessionBytes = [:]
        for (sessionID, list) in events {
            let cost = list.reduce(0) { $0 + $1.storageBytes }
            sessionBytes[sessionID] = cost
            if let taskID = sessions[sessionID]?.taskID {
                bytes[taskID] = (bytes[taskID] ?? 0) + cost
            }
        }
        // A journal that grew past the budget while a smaller one was
        // configured is trimmed on the way in, not left to be discovered
        // later by an allocation failure.
        for taskID in Set(sessions.values.map(\.taskID)) {
            enforceByteBudget(taskID: taskID)
        }
    }

    /// Drop all but the newest `keeping` sessions of a task, oldest first.
    ///
    /// Retention, not editing: the log is append-only within a session's
    /// lifetime, and a caller that wants old sessions gone has to say so.
    /// Returns the sessions that were removed, so the caller can compact the
    /// storage that still holds them.
    @discardableResult
    public func pruneSessions(taskID: UUID, keeping: Int) -> [UUID] {
        guard keeping >= 0 else { return [] }
        let ordered = (sessionsByTask[taskID] ?? [])
        guard ordered.count > keeping else { return [] }
        let doomed = Array(ordered.prefix(ordered.count - keeping))
        for id in doomed { drop(sessionID: id, taskID: taskID) }
        return doomed
    }

    /// Drop the oldest events of a session beyond a count of turns.
    ///
    /// A turn is a prompt and the reply to it. Session boundary events are
    /// never dropped, because a session with no start is not a session.
    @discardableResult
    public func pruneTurns(sessionID: UUID, keeping: Int) -> Int {
        guard keeping >= 0, let existing = events[sessionID] else { return 0 }
        let folded = Self.fold(existing)
        var turnStarts: [Int] = []
        for (offset, event) in folded.enumerated() where event.kind == .userPrompt {
            turnStarts.append(offset)
        }
        guard turnStarts.count > keeping else { return 0 }
        let cut = turnStarts[turnStarts.count - keeping]
        var kept: [SessionEvent] = []
        for (offset, event) in folded.enumerated() {
            if offset >= cut || event.kind == .sessionStarted || event.kind == .sessionEnded {
                kept.append(event)
            }
        }
        let removed = folded.count - kept.count
        events[sessionID] = kept
        let retained = kept.reduce(0) { $0 + $1.storageBytes }
        let previous = sessionBytes[sessionID] ?? 0
        sessionBytes[sessionID] = retained
        if let taskID = sessions[sessionID]?.taskID {
            bytes[taskID] = max(0, (bytes[taskID] ?? 0) - previous + retained)
        }
        return removed
    }

    public func forget(taskID: UUID) {
        for sessionID in sessionsByTask[taskID] ?? [] {
            events[sessionID] = nil
            sessions[sessionID] = nil
            sessionBytes[sessionID] = nil
        }
        sessionsByTask[taskID] = nil
        bytes[taskID] = nil
        tasks[taskID] = nil
    }

    /// Bytes this task's log holds in memory.
    public func byteCount(taskID: UUID) -> Int { bytes[taskID] ?? 0 }

    /// Task identifiers, unsorted. For callers counting bytes on a hot path,
    /// where `tasks()` sorting by creation date is pure waste.
    public func taskIdentifiers() -> [UUID] { Array(tasks.keys) }

    // MARK: - Internals

    private func append(_ event: SessionEvent) async {
        events[event.sessionID, default: []].append(event)
        let cost = event.storageBytes
        bytes[event.taskID] = (bytes[event.taskID] ?? 0) + cost
        sessionBytes[event.sessionID] = (sessionBytes[event.sessionID] ?? 0) + cost
        enforceByteBudget(taskID: event.taskID)
        if let observer { await observer(event) }
    }

    /// Drops the oldest sessions of a task until it is inside its budget.
    ///
    /// The log is RAM-primary and lives in the same process as a model that
    /// wants every byte, so it cannot be allowed to grow with the length of a
    /// project. Dropping the oldest whole session is the right unit: half a
    /// session in memory is worse than none, and the journal file still has
    /// everything for anyone reading back offline.
    ///
    /// The newest session is never dropped, even when a single session
    /// exceeds the budget on its own. Evicting the conversation that is
    /// currently happening would be the one eviction nobody could tolerate.
    private func enforceByteBudget(taskID: UUID) {
        guard options.maxBytesPerTask > 0 else { return }
        while (bytes[taskID] ?? 0) > options.maxBytesPerTask,
              let ordered = sessionsByTask[taskID], ordered.count > 1 {
            drop(sessionID: ordered[0], taskID: taskID)
        }
    }

    private func drop(sessionID: UUID, taskID: UUID) {
        let cost = sessionBytes[sessionID] ?? 0
        bytes[taskID] = max(0, (bytes[taskID] ?? 0) - cost)
        sessionBytes[sessionID] = nil
        events[sessionID] = nil
        sessions[sessionID] = nil
        sessionsByTask[taskID]?.removeAll { $0 == sessionID }
        openResponses = openResponses.filter { $0.value.sessionID != sessionID }
    }

    private func requireSession(_ id: UUID) throws -> Session {
        guard let session = sessions[id] else { throw ContinuityError.unknownSession(id) }
        return session
    }

    private func requireOpenSession(_ id: UUID) throws -> Session {
        let session = try requireSession(id)
        guard session.isOpen else { throw ContinuityError.sessionAlreadyEnded(id) }
        return session
    }

    private struct OpenResponse {
        let sessionID: UUID
        let taskID: UUID
        let model: String?
        let requestID: String?
        let startedAt: Date
        var buffer: String = ""
        var chunkCount: Int = 0
    }
}

public struct SessionLogOptions: Sendable, Equatable {
    /// Bytes of log a task may hold in memory. Zero disables the bound.
    ///
    /// The oldest whole sessions are dropped from memory when it is exceeded.
    /// They remain in the journal file: this bounds what the process holds,
    /// not what was recorded.
    public var maxBytesPerTask: Int

    /// Write one event per streamed chunk in addition to the completed reply.
    ///
    /// Off by default. It exists for callers who need a partial reply to
    /// survive a crash mid-stream; it costs one event per chunk and readers
    /// must fold, which `transcript` and `turns` already do.
    public var persistsChunks: Bool

    public init(persistsChunks: Bool = false,
                maxBytesPerTask: Int = 192 << 20) {
        self.persistsChunks = persistsChunks
        self.maxBytesPerTask = maxBytesPerTask
    }
}

/// One exchange. The response is absent when a reply never arrived, which is
/// itself worth seeing.
public struct SessionTurn: Sendable, Equatable {
    public let sessionID: UUID
    public let promptEventID: UUID?
    public let prompt: String
    public let response: String?
    /// The reply's measurements, when the reply carried any.
    public let responseRecord: ResponseRecord?
    /// When the reply landed. Nil while the turn is unanswered.
    public let completedAt: Date?
    /// When the prompt arrived.
    public let timestamp: Date

    public init(sessionID: UUID,
                promptEventID: UUID?,
                prompt: String,
                response: String?,
                responseRecord: ResponseRecord? = nil,
                completedAt: Date? = nil,
                timestamp: Date) {
        self.sessionID = sessionID
        self.promptEventID = promptEventID
        self.prompt = prompt
        self.response = response
        self.responseRecord = responseRecord
        self.completedAt = completedAt
        self.timestamp = timestamp
    }
}

public struct SessionLogSnapshot: Codable, Sendable, Equatable {
    public var tasks: [ContinuityTask]
    public var sessions: [Session]
    public var events: [SessionEvent]

    public init(tasks: [ContinuityTask] = [],
                sessions: [Session] = [],
                events: [SessionEvent] = []) {
        self.tasks = tasks
        self.sessions = sessions
        self.events = events
    }
}

extension SessionEvent {
    func withResponseID(_ id: UUID) -> SessionEvent {
        SessionEvent(id: self.id, sessionID: sessionID, taskID: taskID,
                     timestamp: timestamp, kind: kind, payload: payload, responseID: id)
    }
}
