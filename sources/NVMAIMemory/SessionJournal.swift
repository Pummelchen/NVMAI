import Foundation

/// The second store: what actually happened, written by the engine.
///
/// Curated memory is the model's, written deliberately, retrieved on demand.
/// The journal is the engine's, written for every turn at no token cost, and
/// never injected into context. They are separate because they answer
/// different questions: "what do I know about this repository" against "what
/// happened in this workspace last Tuesday".
///
/// Content is substance only. A turn is the user's prompt and the assistant's
/// reply text, plus metadata the engine already has. Excluded on purpose:
/// tool definitions, tool-call blocks, tool results, file dumps and command
/// output. That exclusion is what keeps a turn at a few kilobytes, which is
/// what lets the journal live in Valkey rather than needing a file format and
/// an index beside it; it also removes most of what would otherwise have to
/// be redacted.
public protocol SessionJournal: Sendable {
    /// Records one completed turn. Never throws at the caller: a journal that
    /// can fail a completion is worse than no journal.
    func record(_ turn: JournalTurn, in scope: MemoryScope) async
    /// Turns for a session, newest first.
    func turns(session: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn]
    /// Sessions seen in a workspace, newest first.
    func sessions(limit: Int, in scope: MemoryScope) async -> [JournalSessionSummary]
    /// Turns across sessions whose text matches, newest first. For a person
    /// or a tool asking "when did we last touch this", not for the prompt.
    func search(_ text: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn]
}

/// One exchange, as the journal keeps it.
public struct JournalTurn: Sendable, Codable, Equatable {
    public let session: String
    public let workspace: String
    public let index: Int
    public let timestamp: Date
    /// The user's message, filtered to substance.
    public let prompt: String
    /// The assistant's reply text. Tool-call blocks are not part of this.
    public let reply: String
    public let model: String?
    public let promptTokens: Int
    public let completionTokens: Int
    public let latencyMilliseconds: Int
    public let stopReason: String?
    /// Set when the filter dropped material, so a reader knows the turn is a
    /// summary of a larger exchange rather than the whole of a small one.
    public let droppedBytes: Int

    public init(session: String,
                workspace: String,
                index: Int,
                timestamp: Date = Date(),
                prompt: String,
                reply: String,
                model: String? = nil,
                promptTokens: Int = 0,
                completionTokens: Int = 0,
                latencyMilliseconds: Int = 0,
                stopReason: String? = nil,
                droppedBytes: Int = 0) {
        self.session = session
        self.workspace = workspace
        self.index = index
        self.timestamp = timestamp
        self.prompt = prompt
        self.reply = reply
        self.model = model
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.latencyMilliseconds = latencyMilliseconds
        self.stopReason = stopReason
        self.droppedBytes = droppedBytes
    }

    /// Roughly what this turn costs to store.
    public var byteCount: Int {
        prompt.utf8.count + reply.utf8.count + 256
    }
}

public struct JournalSessionSummary: Sendable, Codable, Equatable {
    public let session: String
    public let workspace: String
    public let firstSeen: Date
    public let lastSeen: Date
    public let turnCount: Int
    public let model: String?

    public init(session: String, workspace: String, firstSeen: Date, lastSeen: Date,
                turnCount: Int, model: String?) {
        self.session = session
        self.workspace = workspace
        self.firstSeen = firstSeen
        self.lastSeen = lastSeen
        self.turnCount = turnCount
        self.model = model
    }
}

/// Decides what of a turn is substance.
///
/// The rule is that the journal keeps what a person would want to read back:
/// what was asked and what was answered. Everything the engine can reconstruct
/// or that belongs to the machinery is dropped. Test results survive as the
/// model's statement about them, which is what a later session needs, rather
/// than as thousands of lines of output.
public struct JournalFilter: Sendable, Equatable {
    /// Above this, a message is treated as a dump and summarized to its head
    /// and tail rather than stored whole.
    public var maximumMessageBytes: Int
    /// A fenced block longer than this is replaced by a placeholder naming
    /// its size: a pasted file is context for that turn, not a durable record.
    public var maximumBlockLines: Int

    public init(maximumMessageBytes: Int = 4_096, maximumBlockLines: Int = 12) {
        self.maximumMessageBytes = maximumMessageBytes
        self.maximumBlockLines = maximumBlockLines
    }

    /// Filters one message body. Returns the kept text and how much was
    /// dropped, so the turn can say it is a summary.
    public func filter(_ text: String) -> (kept: String, dropped: Int) {
        let withoutBlocks = collapseLongBlocks(text)
        var kept = withoutBlocks
        var dropped = text.utf8.count - withoutBlocks.utf8.count
        if kept.utf8.count > maximumMessageBytes {
            let head = String(kept.prefix(maximumMessageBytes * 2 / 3))
            let tail = String(kept.suffix(maximumMessageBytes / 4))
            let omitted = kept.utf8.count - head.utf8.count - tail.utf8.count
            kept = head + "\n[... \(omitted) bytes omitted ...]\n" + tail
            dropped += omitted
        }
        return (kept.trimmingCharacters(in: .whitespacesAndNewlines), max(0, dropped))
    }

    /// Replaces long fenced blocks with a one-line placeholder. A short block
    /// is usually the point of the message and stays.
    private func collapseLongBlocks(_ text: String) -> String {
        guard text.contains("```") else { return text }
        var output: [String] = []
        var block: [String] = []
        var fence: String?
        for line in text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init) {
            if line.hasPrefix("```") {
                if fence == nil {
                    fence = line
                    block = []
                } else {
                    if block.count > maximumBlockLines {
                        let bytes = block.reduce(0) { $0 + $1.utf8.count }
                        output.append("[\(block.count) lines, \(bytes) bytes of "
                                      + "\(languageName(fence)) omitted]")
                    } else {
                        output.append(fence ?? "```")
                        output.append(contentsOf: block)
                        output.append(line)
                    }
                    fence = nil
                    block = []
                }
                continue
            }
            if fence != nil { block.append(line) } else { output.append(line) }
        }
        // An unclosed fence: keep what fits under the line limit.
        if fence != nil {
            if block.count > maximumBlockLines {
                output.append("[\(block.count) lines omitted]")
            } else {
                output.append(contentsOf: block)
            }
        }
        return output.joined(separator: "\n")
    }

    private func languageName(_ fence: String?) -> String {
        let name = (fence ?? "").dropFirst(3).trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? "output" : name
    }
}

/// Journal limits. Separate from `MemoryLimits` on purpose: the journal has
/// its own key space and its own trim policy so it can never crowd out
/// curated memory, which is the store with the higher value per byte.
public struct JournalLimits: Sendable, Equatable {
    /// Turns retained per session.
    public var turnsPerSession: Int
    /// Sessions retained per workspace.
    public var sessionsPerWorkspace: Int
    public var filter: JournalFilter

    public init(turnsPerSession: Int = 200,
                sessionsPerWorkspace: Int = 100,
                filter: JournalFilter = .init()) {
        self.turnsPerSession = turnsPerSession
        self.sessionsPerWorkspace = sessionsPerWorkspace
        self.filter = filter
    }

    /// Rough ceiling on what the journal can occupy in a workspace, for the
    /// sizing conversation: turns are 2 to 5 KB with the filter applied.
    public var approximateMaximumBytes: Int {
        turnsPerSession * sessionsPerWorkspace * 4_096
    }
}
