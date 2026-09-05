import Foundation

/// One thing that happened, as the engine saw it.
///
/// Events are append-only and engine-authored. The model is never asked to
/// record its own prompt or reply: the engine already has both, so requiring
/// a tool call for them would spend tokens on information already in hand and
/// would make the record depend on the model choosing to cooperate.
public struct SessionEvent: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public let sessionID: UUID
    public let taskID: UUID
    public let timestamp: Date
    public let kind: SessionEventKind
    public let payload: SessionEventPayload
    /// Ties the pieces of one streamed reply together. It is the identifier
    /// of the `assistantResponseStarted` event, carried by every chunk and by
    /// the completion, so a reader can fold them without guessing from order.
    public let responseID: UUID?

    public init(id: UUID = UUID(),
                sessionID: UUID,
                taskID: UUID,
                timestamp: Date = Date(),
                kind: SessionEventKind,
                payload: SessionEventPayload,
                responseID: UUID? = nil) {
        self.id = id
        self.sessionID = sessionID
        self.taskID = taskID
        self.timestamp = timestamp
        self.kind = kind
        self.payload = payload
        self.responseID = responseID
    }
}

public enum SessionEventKind: String, Codable, Sendable, CaseIterable {
    case sessionStarted
    case userPrompt
    /// A reply that arrives in pieces. Opens a response the chunks belong to.
    case assistantResponseStarted
    case assistantResponseChunk
    case assistantResponseCompleted
    /// A reply recorded in one go, when the caller had it whole.
    case assistantResponse
    case sessionEnded
    /// A memory mutation, so the log can explain a change in belief.
    case memoryWritten
    /// A context handed to a model invocation.
    case contextAssembled

    /// Whether this kind carries part of an assistant reply. The reader folds
    /// these into one logical response.
    public var isResponseFragment: Bool {
        self == .assistantResponseStarted || self == .assistantResponseChunk
            || self == .assistantResponseCompleted
    }
}

/// What an event carries.
///
/// A closed enum rather than a dictionary: every payload shape is known to
/// the engine, and a typed payload cannot silently lose a field the way an
/// untyped bag does.
public enum SessionEventPayload: Codable, Sendable, Equatable {
    case none
    case text(String)
    /// A reply, with whatever the engine happened to know about it.
    case response(ResponseRecord)
    /// A reference to a memory item and the version the write produced.
    case memory(namespace: String, key: String, version: Int, itemID: UUID)
    case context(snapshotID: UUID, itemCount: Int, estimatedTokens: Int)

    public var text: String? {
        switch self {
        case .text(let value): return value
        case .response(let record): return record.text
        case .none, .memory, .context: return nil
        }
    }
}

/// A model reply and its measurements.
///
/// Every measurement is optional because a local engine, a hosted API and a
/// replayed transcript expose different things, and a required field would
/// force callers to invent values.
public struct ResponseRecord: Codable, Sendable, Equatable {
    public var text: String
    public var model: String?
    public var requestID: String?
    public var responseID: String?
    public var inputTokens: Int?
    public var outputTokens: Int?
    public var latencyMilliseconds: Int?
    public var finishReason: String?

    public init(text: String,
                model: String? = nil,
                requestID: String? = nil,
                responseID: String? = nil,
                inputTokens: Int? = nil,
                outputTokens: Int? = nil,
                latencyMilliseconds: Int? = nil,
                finishReason: String? = nil) {
        self.text = text
        self.model = model
        self.requestID = requestID
        self.responseID = responseID
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.latencyMilliseconds = latencyMilliseconds
        self.finishReason = finishReason
    }
}
