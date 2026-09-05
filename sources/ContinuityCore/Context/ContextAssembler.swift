import Foundation

/// Everything an assembler is allowed to see.
///
/// The engine does the fetching and hands over plain values, so an assembler
/// is a pure function of its input. That makes assembly strategies testable
/// without standing up actors, and it stops a custom assembler from reaching
/// into another task's state, because nothing from another task is in here.
public struct ContextRequest: Sendable {
    public let task: ContinuityTask
    public let sessionID: UUID?
    /// Candidate items, already filtered to the eligible ones.
    public let items: [MemoryItem]
    /// Address to item, for resolving dependencies that fall outside the
    /// candidate list.
    public let index: [String: MemoryItem]
    /// Recent exchanges, oldest first.
    public let turns: [SessionTurn]
    public let budget: ContextBudget
    /// The request being answered, when there is one. Assemblers may use it
    /// to bias ranking; the default one uses it for a light keyword lift.
    public let focus: String?
    public let now: Date

    public init(task: ContinuityTask,
                sessionID: UUID? = nil,
                items: [MemoryItem],
                index: [String: MemoryItem] = [:],
                turns: [SessionTurn] = [],
                budget: ContextBudget = ContextBudget(),
                focus: String? = nil,
                now: Date = Date()) {
        self.task = task
        self.sessionID = sessionID
        self.items = items
        self.index = index
        self.turns = turns
        self.budget = budget
        self.focus = focus
        self.now = now
    }
}

/// Turns accumulated state into the block of text a model actually receives.
///
/// A protocol because the right selection differs by domain: a novel wants
/// continuity of character and place, a codebase wants decisions and
/// constraints, a research project wants sources. The engine does not choose;
/// it supplies the state and renders whatever comes back.
public protocol ContextAssembler: Sendable {
    func assemble(_ request: ContextRequest) throws -> ContextSnapshot
}

/// The assembler used unless a caller supplies another.
///
/// Priority order, then importance, then recency. Selected items drag their
/// dependencies in with them. Recent turns get a capped share so a long last
/// exchange cannot displace the durable state, which is the failure mode that
/// makes naive history-window systems lose the plot on chapter 60.
public struct DefaultContextAssembler: ContextAssembler {
    /// Included above the objective, when set. Intended for a short standing
    /// instruction about how to use the state below it.
    public var preamble: String?

    public init(preamble: String? = nil) {
        self.preamble = preamble
    }

    public func assemble(_ request: ContextRequest) throws -> ContextSnapshot {
        let budget = request.budget
        var remaining = budget.maxTokens

        var sections: [String] = []
        var header = "# \(request.task.title)"
        if let preamble, !preamble.isEmpty {
            header += "\n\n\(preamble)"
        }
        if !request.task.objective.isEmpty {
            header += "\n\n## Objective\n\(request.task.objective)"
        }
        remaining -= Self.estimateTokens(header)
        sections.append(header)

        // Recent activity first in the budget, last in the rendering: it is
        // capped, so taking it up front cannot starve the state below.
        var turnBlock: String?
        if budget.recentTurnCount > 0, !request.turns.isEmpty {
            let (text, cost) = renderTurns(request.turns, budget: budget,
                                           allowance: min(budget.turnTokenAllowance,
                                                          max(0, remaining)))
            if let text {
                turnBlock = text
                remaining -= cost
            }
        }

        let ranked = rank(request.items, budget: budget, focus: request.focus)
        var selected: [MemoryItem] = []
        var selectedAddresses = Set<String>()
        var dropped: [MemoryItem] = []

        for item in ranked {
            guard !selectedAddresses.contains(item.address) else { continue }
            var group = [item]
            if budget.includesDependencies {
                group.append(contentsOf: resolveDependencies(of: item,
                                                             index: request.index,
                                                             excluding: selectedAddresses))
            }
            let cost = group.reduce(0) { $0 + $1.estimatedTokens }
            guard cost <= remaining else {
                dropped.append(item)
                continue
            }
            remaining -= cost
            for member in group {
                selected.append(member)
                selectedAddresses.insert(member.address)
            }
        }

        if !selected.isEmpty {
            sections.append(renderItems(selected))
        }
        if let turnBlock {
            sections.append(turnBlock)
        }

        let rendered = sections.joined(separator: "\n\n")
        var versions: [String: Int] = [:]
        for item in selected { versions[item.address] = item.version }

        return ContextSnapshot(taskID: request.task.id,
                               sessionID: request.sessionID,
                               createdAt: request.now,
                               memoryItemIDs: selected.map(\.id),
                               memoryVersions: versions,
                               droppedItemIDs: dropped.map(\.id),
                               renderedContext: rendered,
                               estimatedTokenCount: Self.estimateTokens(rendered),
                               budget: ContextBudgetRecord(budget))
    }

    // MARK: - Ranking

    private func rank(_ items: [MemoryItem],
                      budget: ContextBudget,
                      focus: String?) -> [MemoryItem] {
        let terms = Self.terms(in: focus)
        return items.sorted { lhs, rhs in
            let leftPriority = budget.priority(of: lhs.namespace)
            let rightPriority = budget.priority(of: rhs.namespace)
            if leftPriority != rightPriority { return leftPriority < rightPriority }

            // A disputed fact outranks a settled one at equal priority: an
            // unresolved contradiction is exactly what the model must see.
            let leftDisputed = lhs.status == .disputed
            let rightDisputed = rhs.status == .disputed
            if leftDisputed != rightDisputed { return leftDisputed }

            let leftScore = score(lhs, terms: terms)
            let rightScore = score(rhs, terms: terms)
            if leftScore != rightScore { return leftScore > rightScore }
            if lhs.updatedAt != rhs.updatedAt { return lhs.updatedAt > rhs.updatedAt }
            return lhs.address < rhs.address
        }
    }

    /// Importance, lifted a little when the item mentions what is being
    /// asked. Kept small on purpose: keyword overlap is a weak signal and
    /// should not outrank an explicit importance the caller set.
    private func score(_ item: MemoryItem, terms: Set<String>) -> Double {
        var value = item.importance ?? 0.5
        guard !terms.isEmpty else { return value }
        let haystack = "\(item.address) \(item.value)".lowercased()
        let hits = terms.filter { haystack.contains($0) }.count
        if hits > 0 {
            value += min(0.25, 0.05 * Double(hits))
        }
        return value
    }

    private static func terms(in focus: String?) -> Set<String> {
        guard let focus else { return [] }
        let words = focus.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber && $0 != "_" })
            .map(String.init)
            .filter { $0.count >= 4 }
        return Set(words.prefix(32))
    }

    private func resolveDependencies(of item: MemoryItem,
                                     index: [String: MemoryItem],
                                     excluding: Set<String>) -> [MemoryItem] {
        var seen = excluding
        seen.insert(item.address)
        var out: [MemoryItem] = []
        var frontier = [item]
        var depth = 0
        while !frontier.isEmpty && depth < 3 {
            var next: [MemoryItem] = []
            for current in frontier {
                for address in current.dependencies where !seen.contains(address) {
                    seen.insert(address)
                    guard let resolved = index[address],
                          resolved.status.isEligibleForContext else { continue }
                    out.append(resolved)
                    next.append(resolved)
                }
            }
            frontier = next
            depth += 1
        }
        return out
    }

    // MARK: - Rendering

    private func renderItems(_ items: [MemoryItem]) -> String {
        var lines = ["## Established state"]
        let grouped = Dictionary(grouping: items, by: \.namespace)
        for namespace in grouped.keys.sorted() {
            lines.append("")
            lines.append("### \(namespace)")
            for item in grouped[namespace]!.sorted(by: { $0.key < $1.key }) {
                let marker = item.status == .disputed ? " [disputed]" : ""
                lines.append("- \(item.key) (v\(item.version))\(marker): \(flatten(item.value))")
            }
        }
        return lines.joined(separator: "\n")
    }

    private func renderTurns(_ turns: [SessionTurn],
                             budget: ContextBudget,
                             allowance: Int) -> (String?, Int) {
        guard allowance > 0 else { return (nil, 0) }
        var chosen: [String] = []
        var cost = Self.estimateTokens("## Recent activity")
        // Newest first while filling, so the oldest turns are the ones lost.
        for turn in turns.suffix(budget.recentTurnCount).reversed() {
            let available = allowance - cost
            guard available > 8 else { break }
            // Each turn is trimmed to what is left rather than dropped for
            // being too big. One long exchange would otherwise take the whole
            // allowance with it and leave no recent activity at all, which is
            // the opposite of what a cap is for.
            let fieldLimit = min(budget.maxTurnCharacters,
                                 max(1, (available * 4) / 2 - 24))
            var block = "- asked: \(truncate(flatten(turn.prompt), to: fieldLimit))"
            if let response = turn.response {
                block += "\n  replied: \(truncate(flatten(response), to: fieldLimit))"
            }
            let blockCost = Self.estimateTokens(block)
            guard blockCost <= available else { break }
            cost += blockCost
            chosen.insert(block, at: 0)
        }
        guard !chosen.isEmpty else { return (nil, 0) }
        return ((["## Recent activity"] + chosen).joined(separator: "\n"), cost)
    }

    private func flatten(_ value: String) -> String {
        value.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func truncate(_ value: String, to limit: Int) -> String {
        guard limit > 0, value.count > limit else { return value }
        return String(value.prefix(limit)) + "..."
    }

    /// Four characters per token, the usual English approximation. The
    /// assembler needs a bound it can compute without a tokenizer, and a
    /// bound that is slightly pessimistic is the safe direction to be wrong.
    static func estimateTokens(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        return max(1, text.count / 4)
    }
}
