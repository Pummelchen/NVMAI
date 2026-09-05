import Foundation

enum ServerLog {
    /// S32: requests log from concurrent tasks; serialize stderr writes so
    /// lines never interleave.
    private static let writeLock = NSLock()

    static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        let usage = completion.usage
        write("request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "finish=\(completion.finishReason)")
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    /// Strip report for one request (NVMAI_STRIP_CLI_PROMPT on). The reminder
    /// and token counters are the early-warning signal: if a CLI changes its
    /// bloat template, "reminders=" drops to 0 or "prompt=" jumps back to the
    /// thousands, visible here without a model run.
    static func strip(stats: CLIStrip.Stats, promptTokens: Int) {
        write("strip v\(CLIStrip.version) "
            + "system=\(stats.systemDropped) "
            + "developer=\(stats.developerDropped) "
            + "toolRole=\(stats.toolRoleDropped) "
            + "tools=\(stats.toolsDropped) "
            + "toolCalls=\(stats.toolCallsDropped) "
            + "reminders=\(stats.reminderCharsRemoved)chars "
            + "messageFallback=\(stats.emptyMessageFallbacks) "
            + "requestFallback=\(stats.emptyRequestFallback) "
            + "prompt=\(promptTokens)")
    }

    /// Model residency transitions under --lazy-load / --idle-unload-seconds.
    /// Always logged rather than hidden behind a debug env var: a server that
    /// silently dropped several GB is exactly what an operator needs to see in
    /// the log when a later request is unexpectedly slow.
    static func residency(_ transition: String) {
        write("model \(transition)")
    }

    /// Memory subsystem events. Operational only: never a memory's contents,
    /// which can be anything the model chose to write.
    static func memory(_ detail: String) {
        write("memory \(detail)")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.3fs", seconds)
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        writeLock.withLock {
            FileHandle.standardError.write(Data(line.utf8))
        }
    }
}