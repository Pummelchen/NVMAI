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