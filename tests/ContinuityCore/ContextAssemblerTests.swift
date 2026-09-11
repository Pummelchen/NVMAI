import Foundation
import Testing
@testable import ContinuityCore

@Suite struct ContextAssemblerTests {
    private func item(_ namespace: String,
                      _ key: String,
                      _ value: String,
                      importance: Double? = nil,
                      status: MemoryStatus = .active,
                      dependencies: [String] = [],
                      updatedAt: Date = Date(timeIntervalSince1970: 1_000)) -> MemoryItem {
        MemoryItem(taskID: Self.task.id, namespace: namespace, key: key, value: value,
                   createdAt: updatedAt, updatedAt: updatedAt, status: status,
                   importance: importance, dependencies: dependencies)
    }

    private static let task = ContinuityTask(title: "Pong", objective: "Two autoplayers, no input")

    private func request(_ items: [MemoryItem],
                         budget: ContextBudget,
                         turns: [SessionTurn] = [],
                         focus: String? = nil) -> ContextRequest {
        var index: [String: MemoryItem] = [:]
        for item in items { index[item.address] = item }
        return ContextRequest(task: Self.task, items: items, index: index,
                              turns: turns, budget: budget, focus: focus)
    }

    @Test func theObjectiveIsAlwaysPresent() throws {
        let assembler = DefaultContextAssembler()
        let snapshot = try assembler.assemble(request([], budget: ContextBudget(maxTokens: 16)))
        #expect(snapshot.renderedContext.contains("Two autoplayers, no input"))
        #expect(snapshot.renderedContext.contains("# Pong"))
    }

    @Test func priorityNamespacesWinOverImportance() throws {
        let items = [
            item("notes", "aside", "unimportant colour", importance: 1.0),
            item("decision", "paddle_speed", "6 units per frame", importance: 0.1),
        ]
        let budget = ContextBudget(maxTokens: 4096, priorityNamespaces: ["decision"],
                                   recentTurnCount: 0)
        let snapshot = try DefaultContextAssembler().assemble(request(items, budget: budget))
        let ordered = snapshot.memoryItemIDs
        #expect(ordered.first == items[1].id)
    }

    /// `memoryItemIDs` is ordered by *relevance* and the rendering is ordered
    /// by namespace and key, so the two disagree whenever the most relevant item
    /// is not the one the text happens to list first. It used to be documented as
    /// "render order", which is the opposite; this pins the distinction in both
    /// directions.
    @Test func selectionOrderIsNotRenderOrder() throws {
        // "zebra" sorts last in the rendering but wins the selection on
        // importance.
        let ranked = item("zebra", "alpha_key", "most relevant", importance: 1.0)
        let renderedFirst = item("alpha", "zeta_key", "listed first", importance: 0.1)
        let budget = ContextBudget(maxTokens: 4096, recentTurnCount: 0)
        let snapshot = try DefaultContextAssembler().assemble(
            request([renderedFirst, ranked], budget: budget))

        #expect(snapshot.memoryItemIDs == [ranked.id, renderedFirst.id],
                "memoryItemIDs follows the selection, not the text")
        let rendered = snapshot.renderedContext
        let alpha = try #require(rendered.range(of: "### alpha"))
        let zebra = try #require(rendered.range(of: "### zebra"))
        #expect(alpha.lowerBound < zebra.lowerBound,
                "the rendering groups by namespace and sorts it")
        // Same items, so the counts agree; only the order differs.
        #expect(snapshot.memoryItemIDs.count == 2)
    }

    /// The budget is the whole point of the assembler, so it is checked
    /// against the rendered text rather than against the selection.
    @Test func theBudgetIsRespectedAndOverflowIsReported() throws {
        let long = String(repeating: "detail ", count: 200)
        let items = (0..<10).map { index in
            item("state", "k\(index)", long, importance: Double(10 - index) / 10)
        }
        let budget = ContextBudget(maxTokens: 400, recentTurnCount: 0)
        let snapshot = try DefaultContextAssembler().assemble(request(items, budget: budget))

        #expect(snapshot.estimatedTokenCount <= budget.maxTokens)
        #expect(snapshot.memoryItemIDs.isEmpty == false)
        #expect(snapshot.droppedItemIDs.isEmpty == false)
        #expect(snapshot.memoryItemIDs.count + snapshot.droppedItemIDs.count == items.count)
        // The most important item is the one that survives.
        #expect(snapshot.memoryItemIDs.contains(items[0].id))
    }

    @Test func dependenciesComeInWithTheItemThatNeedsThem() throws {
        let constraint = item("constraint", "no_input", "neither paddle reads the mouse",
                              importance: 0.01)
        let decision = item("decision", "autoplay", "both paddles track the ball",
                            importance: 0.9, dependencies: ["constraint.no_input"])
        let filler = (0..<20).map {
            item("filler", "k\($0)", String(repeating: "x", count: 200), importance: 0.5)
        }
        let budget = ContextBudget(maxTokens: 300, priorityNamespaces: ["decision"],
                                   recentTurnCount: 0)
        let snapshot = try DefaultContextAssembler()
            .assemble(request([decision, constraint] + filler, budget: budget))

        #expect(snapshot.memoryItemIDs.contains(decision.id))
        // Without dependency pull-in the constraint's low importance would
        // have put it far below the cut.
        #expect(snapshot.memoryItemIDs.contains(constraint.id))
    }

    @Test func dependenciesCanBeTurnedOff() throws {
        let constraint = item("constraint", "no_input", "no mouse", importance: 0.01)
        let decision = item("decision", "autoplay", "tracks the ball", importance: 0.9,
                            dependencies: ["constraint.no_input"])
        var budget = ContextBudget(maxTokens: 60, priorityNamespaces: ["decision"],
                                   recentTurnCount: 0)
        budget.includesDependencies = false
        let snapshot = try DefaultContextAssembler()
            .assemble(request([decision, constraint], budget: budget))
        #expect(snapshot.memoryItemIDs.contains(decision.id))
    }

    @Test func recentTurnsAreCappedSoStateSurvives() throws {
        let wall = String(repeating: "chatter ", count: 500)
        let turns = (0..<6).map { index in
            SessionTurn(sessionID: UUID(), promptEventID: nil,
                        prompt: "ask \(index)", response: wall,
                        timestamp: Date(timeIntervalSince1970: TimeInterval(index)))
        }
        let facts = (0..<5).map { item("decision", "k\($0)", "value \($0)", importance: 0.9) }
        let budget = ContextBudget(maxTokens: 500, priorityNamespaces: ["decision"],
                                   recentTurnCount: 6, turnShare: 0.3)
        let snapshot = try DefaultContextAssembler()
            .assemble(request(facts, budget: budget, turns: turns))

        #expect(snapshot.estimatedTokenCount <= budget.maxTokens)
        // Every fact fits even though the transcript alone would have
        // exhausted the window several times over.
        #expect(snapshot.memoryItemIDs.count == facts.count)
        #expect(snapshot.renderedContext.contains("Recent activity"))
        // The newest exchange is the one kept, trimmed to fit rather than
        // dropped for being long.
        #expect(snapshot.renderedContext.contains("ask 5"))
        #expect(snapshot.renderedContext.contains("ask 0") == false)
    }

    @Test func turnsAreExcludedWhenTheBudgetSaysStateOnly() throws {
        let turns = [SessionTurn(sessionID: UUID(), promptEventID: nil,
                                 prompt: "hello", response: "hi",
                                 timestamp: Date())]
        let snapshot = try DefaultContextAssembler()
            .assemble(request([item("n", "k", "v")],
                              budget: .stateOnly(maxTokens: 1000), turns: turns))
        #expect(snapshot.renderedContext.contains("Recent activity") == false)
    }

    @Test func disputedItemsAreFlaggedAndRankedUp() throws {
        let settled = item("fact", "a_settled", "agreed", importance: 0.9)
        let disputed = item("fact", "z_disputed", "contested", importance: 0.9,
                            status: .disputed)
        let snapshot = try DefaultContextAssembler()
            .assemble(request([settled, disputed],
                              budget: ContextBudget(maxTokens: 4096, recentTurnCount: 0)))
        #expect(snapshot.memoryItemIDs.first == disputed.id)
        #expect(snapshot.renderedContext.contains("[disputed]"))
    }

    @Test func focusLiftsItemsThatMentionTheRequest() throws {
        let paddle = item("notes", "paddle", "paddle speed is six", importance: 0.5)
        let colour = item("notes", "colour", "the background is black", importance: 0.5)
        let snapshot = try DefaultContextAssembler()
            .assemble(request([colour, paddle],
                              budget: ContextBudget(maxTokens: 4096, recentTurnCount: 0),
                              focus: "how fast should the paddle move"))
        #expect(snapshot.memoryItemIDs.first == paddle.id)
    }

    @Test func snapshotsRecordVersionsSoTheyCanBeReplayed() throws {
        var versioned = item("decision", "storage", "native swift")
        versioned.version = 7
        let snapshot = try DefaultContextAssembler()
            .assemble(request([versioned],
                              budget: ContextBudget(maxTokens: 4096, recentTurnCount: 0)))
        #expect(snapshot.memoryVersions["decision.storage"] == 7)
        #expect(snapshot.renderedContext.contains("(v7)"))
        #expect(snapshot.budget.maxTokens == 4096)
    }

    @Test func assemblyIsDeterministic() throws {
        let items = (0..<12).map {
            item("n\($0 % 3)", "k\($0)", "value \($0)", importance: Double($0 % 5) / 5)
        }
        let budget = ContextBudget(maxTokens: 300, priorityNamespaces: ["n1"],
                                   recentTurnCount: 0)
        let assembler = DefaultContextAssembler()
        let first = try assembler.assemble(request(items, budget: budget))
        let second = try assembler.assemble(request(items, budget: budget))
        #expect(first.renderedContext == second.renderedContext)
        #expect(first.memoryItemIDs == second.memoryItemIDs)
    }

    @Test func preambleIsRenderedBeforeTheObjective() throws {
        let assembler = DefaultContextAssembler(preamble: "Use the state below as fact.")
        let snapshot = try assembler.assemble(request([], budget: ContextBudget()))
        let text = snapshot.renderedContext
        let preambleIndex = text.range(of: "Use the state below")?.lowerBound
        let objectiveIndex = text.range(of: "## Objective")?.lowerBound
        #expect(preambleIndex != nil && objectiveIndex != nil)
        #expect(preambleIndex! < objectiveIndex!)
    }
}
