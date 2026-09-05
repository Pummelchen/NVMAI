import Foundation

/// Exactly what was put in front of the model, and what it was built from.
///
/// Reproducibility is the point. A snapshot names the memory items and the
/// version of each, so a reply can be explained months later even after every
/// one of those items has been rewritten. Storing only the rendered string
/// would answer "what did it see" but not "what did it see it as".
public struct ContextSnapshot: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public let taskID: UUID
    public let sessionID: UUID?
    public let createdAt: Date
    /// The included items, in the order they were rendered.
    public let memoryItemIDs: [UUID]
    /// Address to version, for the items above.
    public let memoryVersions: [String: Int]
    /// Items that matched but did not fit. Named, because a persistent
    /// omission is a signal that the budget or the priorities are wrong.
    public let droppedItemIDs: [UUID]
    public let renderedContext: String
    public let estimatedTokenCount: Int
    public let budget: ContextBudgetRecord

    public init(id: UUID = UUID(),
                taskID: UUID,
                sessionID: UUID? = nil,
                createdAt: Date = Date(),
                memoryItemIDs: [UUID],
                memoryVersions: [String: Int],
                droppedItemIDs: [UUID] = [],
                renderedContext: String,
                estimatedTokenCount: Int,
                budget: ContextBudgetRecord) {
        self.id = id
        self.taskID = taskID
        self.sessionID = sessionID
        self.createdAt = createdAt
        self.memoryItemIDs = memoryItemIDs
        self.memoryVersions = memoryVersions
        self.droppedItemIDs = droppedItemIDs
        self.renderedContext = renderedContext
        self.estimatedTokenCount = estimatedTokenCount
        self.budget = budget
    }

    public var isEmpty: Bool { renderedContext.isEmpty }
}

/// The budget as it was at assembly time.
///
/// `ContextBudget` is a live knob callers tune between calls, so a snapshot
/// keeps its own flattened copy. Otherwise the record of an old assembly
/// would silently describe today's settings.
public struct ContextBudgetRecord: Codable, Sendable, Equatable {
    public let maxTokens: Int
    public let priorityNamespaces: [String]
    public let recentTurnCount: Int

    public init(_ budget: ContextBudget) {
        self.maxTokens = budget.maxTokens
        self.priorityNamespaces = budget.priorityNamespaces
        self.recentTurnCount = budget.recentTurnCount
    }
}
