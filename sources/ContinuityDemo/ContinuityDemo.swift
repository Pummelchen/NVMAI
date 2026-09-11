import Foundation
import ContinuityCore

/// A demonstration and a diagnostic.
///
/// It runs the two shapes of work the engine is built for, prints what the
/// model would actually have seen at each step, and then reports what the
/// stores cost. The point is that both scenarios use the same engine: only
/// the namespaces and the budget differ.
///
/// Run it with `swift run ContinuityDemo`, or `swift run ContinuityDemo novel`
/// / `coding` / `diagnose` for one section.

@main
struct ContinuityDemo {
    static func main() async {
        let argument = CommandLine.arguments.dropFirst().first?.lowercased()
        do {
            switch argument {
            case "coding": try await coding()
            case "novel": try await novel()
            case "diagnose":
                try await diagnose(keepingAt: CommandLine.arguments.dropFirst(2).first
                                    .map { URL(fileURLWithPath: $0) })
            case "inspect":
                let path = CommandLine.arguments.dropFirst(2).first
                guard let path else {
                    print("usage: ContinuityDemo inspect <journal.ndjson>")
                    exit(2)
                }
                try inspect(URL(fileURLWithPath: path))
            case nil, "all":
                try await coding()
                print("")
                try await novel()
                print("")
                try await diagnose()
            default:
                print("usage: ContinuityDemo [coding|novel|diagnose|inspect <file>|all]")
                exit(2)
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    // MARK: - A long coding project

    /// Three sessions, weeks apart, on one program. The third session has
    /// none of the first two in its window and still knows the rules.
    static func coding() async throws {
        heading("A coding project across three sessions")

        let engine = ContinuityEngine(
            configuration: ContinuityConfiguration(
                defaultBudget: ContextBudget(
                    maxTokens: 700,
                    priorityNamespaces: ["objective", "constraint", "decision", "gotcha"],
                    recentTurnCount: 2)))
        try await engine.start()

        let task = try await engine.createTask(
            title: "Pong with two autoplayers",
            objective: "A Pong game where both paddles play themselves. "
                + "Swift first, then a Python port, then C99.")

        // Session one: the rules get decided.
        let first = try await engine.beginSession(taskID: task.id, model: "qwen35b")
        try await engine.recordUserPrompt(sessionID: first.id,
                                          text: "Write Pong in Swift with two computer players.")
        try await engine.recordAssistantResponse(
            sessionID: first.id,
            text: "Done. 800x600 field, first to 11, ball starts at 5 and gains 0.5 per "
                + "paddle hit up to 12, paddles move 6 units per frame.")
        try await engine.remember(sessionID: first.id, namespace: "decision", key: "field",
                                  value: "800 by 600, first to 11 points",
                                  importance: 0.9, tags: ["rules"])
        try await engine.remember(sessionID: first.id, namespace: "decision", key: "ball_speed",
                                  value: "starts at 5, +0.5 per paddle hit, capped at 12",
                                  importance: 0.9, tags: ["rules"])
        try await engine.remember(sessionID: first.id, namespace: "decision", key: "paddle_speed",
                                  value: "6 units per frame, 3-unit deadzone to stop jitter",
                                  importance: 0.85, tags: ["rules"],
                                  dependencies: ["constraint.no_human_input"])
        try await engine.remember(sessionID: first.id, namespace: "constraint",
                                  key: "no_human_input",
                                  value: "neither paddle reads the keyboard or mouse; "
                                      + "both track the ball",
                                  importance: 0.5)
        _ = try await engine.endSession(first.id)

        // Session two: a bug becomes a durable gotcha.
        let second = try await engine.beginSession(taskID: task.id, model: "qwen35b")
        try await engine.recordUserPrompt(sessionID: second.id,
                                          text: "The paddles vibrate when the ball is level.")
        try await engine.recordAssistantResponse(
            sessionID: second.id,
            text: "The deadzone was smaller than the paddle step, so it overshot every frame. "
                + "Widened it to 3 units.")
        try await engine.remember(sessionID: second.id, namespace: "gotcha", key: "jitter",
                                  value: "a deadzone smaller than the paddle step makes the "
                                      + "paddle oscillate; keep deadzone >= step / 2",
                                  importance: 0.8)
        _ = try await engine.endSession(second.id)

        // Session three: the port. Nothing above is in its window.
        let third = try await engine.beginSession(taskID: task.id, model: "qwen35b")
        let prompt = "Now port it to C99. Keep the behaviour identical."
        let context = try await engine.assembleContext(taskID: task.id,
                                                       sessionID: third.id,
                                                       focus: prompt)
        print(context.renderedContext)
        print("")
        note("\(context.estimatedTokenCount) estimated tokens, "
             + "\(context.memoryItemIDs.count) facts, "
             + "\(context.droppedItemIDs.count) dropped")
        note("the constraint arrives with the paddle decision that depends on it, "
             + "even though its own importance is low")
    }

    // MARK: - A hundred-chapter novel

    /// The failure this prevents is the one that ships: two chapters that
    /// disagree, sixty chapters apart.
    static func novel() async throws {
        heading("A novel at chapter 41")

        let engine = ContinuityEngine(
            configuration: ContinuityConfiguration(
                defaultBudget: ContextBudget(
                    maxTokens: 600,
                    priorityNamespaces: ["character", "plot", "setting", "style"],
                    recentTurnCount: 0)))
        try await engine.start()

        let task = try await engine.createTask(
            title: "The Photograph",
            objective: "A hundred-chapter novel. Close third person, past tense.")
        let session = try await engine.beginSession(taskID: task.id, model: "qwen35b")

        try await engine.remember(sessionID: session.id, namespace: "style", key: "voice",
                                  value: "close third person, past tense, no head-hopping",
                                  importance: 0.95)
        try await engine.remember(sessionID: session.id, namespace: "character.marcus",
                                  key: "knows_about_photo",
                                  value: "as of chapter 12 Marcus has NOT seen the photograph",
                                  importance: 0.9, tags: ["continuity"])
        try await engine.remember(sessionID: session.id, namespace: "character.marcus",
                                  key: "eyes", value: "grey", importance: 0.4)
        try await engine.remember(sessionID: session.id, namespace: "plot.act2", key: "brother",
                                  value: "the brother is missing; found alive in chapter 58",
                                  importance: 0.9,
                                  dependencies: ["character.marcus.knows_about_photo"])
        try await engine.remember(sessionID: session.id, namespace: "setting", key: "town",
                                  value: "Ashgrove, coastal, permanently out of season",
                                  importance: 0.6)

        // Chapter 30 contradicted chapter 12. The engine does not pick a
        // winner; it shows the model the conflict.
        try await engine.remember(sessionID: session.id, namespace: "character.marcus",
                                  key: "knows_about_photo",
                                  value: "chapter 30 has Marcus recognising the photograph, "
                                      + "which contradicts chapter 12",
                                  importance: 0.95, tags: ["continuity"])
        _ = try await engine.dispute(taskID: task.id, namespace: "character.marcus",
                                     key: "knows_about_photo")

        let prompt = "Write chapter 41, where Marcus finally confronts his brother's absence."
        let context = try await engine.assembleContext(taskID: task.id,
                                                       sessionID: session.id, focus: prompt)
        print(context.renderedContext)
        print("")

        let history = await engine.history(taskID: task.id,
                                           namespace: "character.marcus",
                                           key: "knows_about_photo")
        note("what the book believed about the photograph, in order:")
        for version in history {
            print("      v\(version.version) [\(version.status.rawValue)] \(version.value)")
        }
        note("the contradiction is surfaced, not silently resolved; "
             + "chapter 12's version is still readable")
    }

    // MARK: - Diagnostics

    /// What the stores cost, on a task the size of a real one.
    /// - Parameter keepingAt: write the journal here and leave it behind, so
    ///   it can be handed to `inspect`.
    static func diagnose(keepingAt destination: URL? = nil) async throws {
        heading("Diagnostics")

        let directory = destination?.deletingLastPathComponent()
            ?? FileManager.default.temporaryDirectory
                .appendingPathComponent("continuity-demo-\(UUID().uuidString)")
        // Cleans up at return, not at the end of the `if`: `defer` binds to its
        // enclosing scope, and inside that `if` it ran immediately -- deleting
        // the directory the journal below is then written to.
        defer {
            if destination == nil {
                try? FileManager.default.removeItem(at: directory)
            }
        }
        let url = destination ?? directory.appendingPathComponent("journal.ndjson")

        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let task = try await engine.createTask(title: "Scale check",
                                               objective: "A hundred chapters of state")

        let start = Date()
        for chapter in 0..<100 {
            let session = try await engine.beginSession(taskID: task.id, model: "qwen35b")
            try await engine.recordUserPrompt(sessionID: session.id,
                                              text: "Write chapter \(chapter).")
            try await engine.recordAssistantResponse(
                sessionID: session.id,
                text: String(repeating: "prose ", count: 400))
            try await engine.remember(sessionID: session.id, namespace: "chapter",
                                      key: String(format: "c%03d", chapter),
                                      value: "chapter \(chapter): summary of what happened",
                                      importance: Double(chapter % 10) / 10)
            _ = try await engine.endSession(session.id)
        }
        let elapsed = Date().timeIntervalSince(start)

        let statistics = await engine.statistics()
        let context = try await engine.assembleContext(
            taskID: task.id,
            budget: ContextBudget(maxTokens: 2000, priorityNamespaces: ["chapter"],
                                  recentTurnCount: 2))
        let bytes = (try? FileManager.default
            .attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0

        row("sessions", "\(statistics.sessionCount)")
        row("events", "\(statistics.eventCount)")
        row("memory items", "\(statistics.memoryItemCount)")
        row("journal records", "\(statistics.journaledRecords)")
        row("journal bytes", "\(bytes)")
        row("memory bytes", "\(statistics.memoryBytes)")
        row("log bytes resident", "\(statistics.logBytes)")
        row("100 sessions in", String(format: "%.0f ms", elapsed * 1000))
        row("per session", String(format: "%.1f ms", elapsed * 10))
        row("assembled context", "\(context.estimatedTokenCount) tokens, "
            + "\(context.memoryItemIDs.count) of \(statistics.memoryItemCount) facts")

        try await engine.compactJournal()
        let compacted = (try? FileManager.default
            .attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        row("after compaction", "\(compacted) bytes")
        await engine.shutDown()
        if let destination {
            note("journal kept at \(destination.path)")
        }
        note("the context stays inside its budget while the record keeps everything; "
             + "that separation is the whole design")
        note("per-session cost is dominated by the durability barrier taken at each "
             + "session boundary, which is the point of taking it there and not per write")
    }

    // MARK: - Inspecting a journal

    /// Read a journal file without opening it for writing.
    ///
    /// A store that only its own process can look at is a store nobody can
    /// debug. This is deliberately read-only: it takes no lock, so it is safe
    /// to point at the file of a server that is running.
    static func inspect(_ url: URL) throws {
        guard let contents = FileManager.default.contents(atPath: url.path) else {
            print("no journal at \(url.path)")
            exit(1)
        }
        let records = try FileJournal.read(contentsOf: url)

        heading("Journal at \(url.lastPathComponent)")
        var tasks: [UUID: ContinuityTask] = [:]
        var sessions: [UUID: Session] = [:]
        var items: [String: MemoryItem] = [:]
        var versions = 0
        var events = 0
        var checkpoints = 0
        for record in records {
            switch record {
            case .task(let task): tasks[task.id] = task
            case .session(let session): sessions[session.id] = session
            case .event: events += 1
            case .memory(let item): items["\(item.taskID)/\(item.address)"] = item
            case .memoryVersion: versions += 1
            case .checkpoint(let log, let memory):
                checkpoints += 1
                for task in log.tasks { tasks[task.id] = task }
                for session in log.sessions { sessions[session.id] = session }
                for item in memory.items { items["\(item.taskID)/\(item.address)"] = item }
                events += log.events.count
                versions += memory.versions.count
            }
        }

        row("file bytes", "\(contents.count)")
        row("records", "\(records.count)")
        row("checkpoints", "\(checkpoints)")
        row("tasks", "\(tasks.count)")
        row("sessions", "\(sessions.count)")
        row("events", "\(events)")
        row("memory items", "\(items.count)")
        row("retained versions", "\(versions)")
        let bytes = items.values.reduce(0) { $0 + $1.storageBytes }
        row("memory bytes", "\(bytes)")

        for task in tasks.values.sorted(by: { $0.createdAt < $1.createdAt }) {
            print("")
            print("  \(task.title)  [\(task.id)]")
            if !task.objective.isEmpty { print("    objective: \(task.objective)") }
            let taskSessions = sessions.values.filter { $0.taskID == task.id }
                .sorted { $0.startedAt < $1.startedAt }
            for session in taskSessions.suffix(10) {
                let label = session.tag ?? session.externalID ?? "-"
                let state = session.endedAt == nil ? "open" : "ended"
                print("    session \(label) [\(state)] \(session.model ?? "")")
            }
            if taskSessions.count > 10 { print("    ...and \(taskSessions.count - 10) earlier sessions") }
            let mine = items.values.filter { $0.taskID == task.id }
            for item in mine.sorted(by: { $0.address < $1.address }).prefix(40) {
                let marker = item.status == .active ? " " : "\(item.status.rawValue.prefix(1))"
                print("    \(marker) \(item.address) (v\(item.version)): "
                      + "\(summarize(item.value))")
            }
            if mine.count > 40 { print("    ...and \(mine.count - 40) more") }
        }
    }

    static func summarize(_ value: String, limit: Int = 90) -> String {
        let flat = value.replacingOccurrences(of: "\n", with: " ")
        return flat.count > limit ? String(flat.prefix(limit)) + "..." : flat
    }

    // MARK: - Output

    static func heading(_ text: String) {
        print("\u{001B}[1m\(text)\u{001B}[0m")
        print(String(repeating: "-", count: text.count))
        print("")
    }

    static func note(_ text: String) { print("  note: \(text)") }

    static func row(_ label: String, _ value: String) {
        print("  \(label.padding(toLength: 20, withPad: " ", startingAt: 0)) \(value)")
    }
}
