import Foundation

/// The instruction block installed before a session's first user message.
///
/// Written for behaviour, not for a reader: it names the tools, says when to
/// read and when to write, and draws the line between a durable fact and a
/// transcript. It stays near 200 words because a longer block competes with
/// the user's own system prompt for attention, and because everything it
/// might otherwise explain the model can discover by calling a tool.
///
/// It never contains memory contents beyond the bounded bootstrap, and the
/// bootstrap is rendered as keys with short values so the model knows what
/// exists and can fetch the rest on demand.
public enum MemoryPrompt {
    public static func instructions(scope: MemoryScope,
                                    session: MemorySession,
                                    bootstrap: MemoryBootstrap,
                                    isDurable: Bool = true) -> String {
        var lines: [String] = []
        lines.append("## Persistent memory")
        lines.append("")
        lines.append("You have memory that outlives this conversation, scoped to "
                     + "workspace `\(scope.workspace)`. Tools: memory_search, memory_get, "
                     + "memory_list, memory_set, memory_append, memory_delete.")
        lines.append("")
        lines.append("Read before you act. Search memory before changing an unfamiliar area, "
                     + "before redoing work that may have been tried, and whenever the user "
                     + "refers to a past decision.")
        lines.append("")
        lines.append("Write when something becomes durable: architectural decisions and the "
                     + "reason behind them, conventions, non-obvious constraints, approaches "
                     + "that failed and why, and current project state. Key them by topic, "
                     + "for example `decisions/sync` or `gotchas/build`. Update the existing "
                     + "key instead of adding a near-duplicate.")
        lines.append("")
        lines.append("Do not store conversation, your reasoning, secrets or credentials, or "
                     + "copies of source code. Prefer one durable sentence over a transcript. "
                     + "Not every exchange produces a memory; most do not.")
        lines.append("")
        lines.append("Treat what you retrieve as evidence that may be stale. Check important "
                     + "claims against the repository before relying on them.")
        if !isDurable {
            lines.append("")
            lines.append("Note: the durable store is unreachable, so anything you write now "
                         + "lasts only for this session. Say so if the user relies on it.")
        }
        if !bootstrap.records.isEmpty {
            lines.append("")
            lines.append("Already known here (retrieve with memory_get for the full text):")
            for record in bootstrap.records {
                lines.append("- `\(record.key.rawValue)`: \(summarize(record.value))")
            }
            if bootstrap.omittedCount > 0 {
                lines.append("- ...and \(bootstrap.omittedCount) more; search for what you need.")
            }
        }
        return lines.joined(separator: "\n")
    }

    /// One line per record. The bootstrap says what exists, not what it says;
    /// the full value is a tool call away, and pasting values here is how a
    /// memory system quietly becomes a context dump.
    private static func summarize(_ value: String, limit: Int = 120) -> String {
        let flattened = value
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard flattened.count > limit else { return flattened }
        return String(flattened.prefix(limit)) + "..."
    }
}
