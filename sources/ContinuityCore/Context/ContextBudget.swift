import Foundation

/// How much context a call may spend, and what to spend it on first.
///
/// A budget is not a nicety. On a hundred-chapter book the accumulated state
/// exceeds any window within a few chapters, so something must be dropped on
/// every single call. Making that choice explicit and ordered is the
/// difference between a system that forgets the least important thing and one
/// that forgets whatever happened to sort last.
public struct ContextBudget: Sendable, Equatable {
    /// Ceiling for the whole assembled block, in estimated tokens.
    public var maxTokens: Int
    /// Namespaces in descending priority. A namespace listed here outranks
    /// every unlisted one, and prefixes match, so `constraint` covers
    /// `constraint.character`.
    public var priorityNamespaces: [String]
    /// How many recent prompt-and-reply pairs to include.
    public var recentTurnCount: Int
    /// Share of the budget recent turns may take, 0...1. The remainder goes
    /// to memory. Turns are capped rather than trimmed last because a long
    /// recent exchange would otherwise crowd out every durable fact.
    public var turnShare: Double
    /// Longest single turn excerpt, in characters.
    public var maxTurnCharacters: Int
    /// Pull in the items an included item depends on, even when they rank
    /// below the cut. A decision without its constraint invites the model to
    /// re-derive the decision wrongly.
    public var includesDependencies: Bool

    public init(maxTokens: Int = 4096,
                priorityNamespaces: [String] = [],
                recentTurnCount: Int = 4,
                turnShare: Double = 0.35,
                maxTurnCharacters: Int = 1200,
                includesDependencies: Bool = true) {
        self.maxTokens = max(0, maxTokens)
        self.priorityNamespaces = priorityNamespaces
        self.recentTurnCount = max(0, recentTurnCount)
        self.turnShare = min(max(turnShare, 0), 1)
        self.maxTurnCharacters = max(0, maxTurnCharacters)
        self.includesDependencies = includesDependencies
    }

    /// State only, no transcript. What a fresh session on a long task wants:
    /// the accumulated facts, not the last conversation's small talk.
    public static func stateOnly(maxTokens: Int = 4096,
                                 priorityNamespaces: [String] = []) -> ContextBudget {
        ContextBudget(maxTokens: maxTokens,
                      priorityNamespaces: priorityNamespaces,
                      recentTurnCount: 0,
                      turnShare: 0)
    }

    /// Rank for a namespace, lower is more important.
    func priority(of namespace: String) -> Int {
        for (index, prefix) in priorityNamespaces.enumerated() {
            if namespace == prefix || namespace.hasPrefix(prefix + ".") { return index }
        }
        return priorityNamespaces.count
    }

    var turnTokenAllowance: Int {
        Int(Double(maxTokens) * turnShare)
    }
}
