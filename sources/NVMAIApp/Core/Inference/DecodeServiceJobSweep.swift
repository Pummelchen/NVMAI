import Foundation

/// Finds launchd jobs a previous run of this app left behind.
///
/// `DecodeServiceInferenceClient.tearDownService` removes the job it started,
/// but a force-quit or a crash never reaches it. The job stays bootstrapped, so
/// its helper keeps the model resident -- about 20 GB -- and the next launch
/// bootstraps a *second* job under a fresh pid+token label. Nothing looked for
/// the first one: `ensureProcess` only ever asks about its own label, and the
/// label is built fresh from `getpid()`.
///
/// The label is the only identity such a job carries:
/// `com.nvmai.decode.<uid>.<owner-pid>.<token>`. That makes the rule simple and
/// safe -- a job under this prefix whose owning pid is gone is an orphan, while
/// a job whose owner is alive belongs to a running app (possibly another
/// checkout's build of it, or an app the user still has open) and is left
/// alone. A pid the kernel has since reused reads as alive, which errs towards
/// leaving an orphan in place rather than killing a stranger's helper.
enum DecodeServiceJobSweep {
    static let labelPrefix = "com.nvmai.decode."

    struct Job: Equatable {
        /// The launchd label, which is also what `launchctl bootout` takes.
        let label: String
        /// The pid of the app that bootstrapped the job, read from the label.
        let owner: pid_t
        /// The socket file's *name* under the socket directory,
        /// `<owner>.<token>.sock`. Kept as a name rather than a path so the
        /// pure part of the sweep never touches the filesystem.
        let socketName: String
    }

    /// Parses `launchctl list`. The table is `PID<TAB>Status<TAB>Label`, one
    /// job per line, with a header; a loaded-but-not-running job shows `-` for
    /// its pid, which says nothing about its owner, so only the label is read.
    /// Anything that does not parse as one of our labels is skipped rather than
    /// guessed at, and the uid filter keeps another user's jobs out of reach
    /// even if a label happens to look like ours.
    static func parseLaunchctlList(_ output: String, uid: uid_t) -> [Job] {
        output.split(separator: "\n").compactMap { line in
            // The label is the last field and contains no whitespace; taking the
            // last field that carries the prefix tolerates a changed column set.
            guard let field = line.split(whereSeparator: \.isWhitespace)
                    .last(where: { $0.hasPrefix(labelPrefix) }) else { return nil }
            let label = String(field)
            let parts = label.split(separator: ".", omittingEmptySubsequences: false)
            guard parts.count == 6,
                  parts[0] == "com", parts[1] == "nvmai", parts[2] == "decode",
                  parts[3] == String(uid),
                  let owner = pid_t(parts[4]),
                  !parts[5].isEmpty else { return nil }
            return Job(label: label,
                       owner: owner,
                       socketName: "\(parts[4]).\(parts[5]).sock")
        }
    }

    /// The subset to remove: our jobs whose owning app is no longer running.
    static func orphans(in jobs: [Job], isAlive: (pid_t) -> Bool) -> [Job] {
        jobs.filter { !isAlive($0.owner) }
    }

    /// Whether a pid names a live process. `EPERM` means it exists but belongs
    /// to someone else, which still counts as alive -- the sweep must not kill a
    /// helper it cannot even signal.
    static func processIsAlive(_ pid: pid_t) -> Bool {
        guard pid > 0 else { return false }
        if kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }
}
