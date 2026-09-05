import Foundation

/// Where a piece of remembered state came from.
///
/// Provenance is not decoration. When a long task goes wrong the question is
/// always "what did the system believe when it produced that, and who told it
/// so", and answering it needs a chain from a memory version back to the
/// session and the event that caused it. Recording that at write time is the
/// only moment it is cheap; reconstructing it afterwards is impossible.
public struct Provenance: Codable, Sendable, Equatable {
    /// The session that caused the write, when a session caused it.
    public let sessionID: UUID?
    /// The specific event, usually an assistant response, behind the write.
    public let eventID: UUID?
    /// Who or what performed the write.
    public let author: ProvenanceAuthor
    public let timestamp: Date

    public init(sessionID: UUID? = nil,
                eventID: UUID? = nil,
                author: ProvenanceAuthor = .engine,
                timestamp: Date = Date()) {
        self.sessionID = sessionID
        self.eventID = eventID
        self.author = author
        self.timestamp = timestamp
    }
}

/// Distinguishing a fact the model asserted from one a person stated matters
/// when they disagree: a user instruction outranks a model inference, and
/// without the author the two are indistinguishable after the fact.
public enum ProvenanceAuthor: String, Codable, Sendable, CaseIterable {
    /// Written by the model through the memory API.
    case model
    /// Written because a person said so.
    case user
    /// Written by the engine itself, such as a task's opening objective.
    case engine
    /// Written by an offline analysis pass over the session log.
    case extractor
}
