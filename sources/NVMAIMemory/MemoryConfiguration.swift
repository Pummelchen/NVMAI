import Foundation

/// Where continuity state lives on this machine.
///
/// There is no host and no port. The store runs inside the server process,
/// so the only things left to configure are the directory it writes to and
/// how much it is allowed to hold.
public struct ContinuityStorageConfiguration: Sendable, Equatable {
    /// Directory holding the journals. Created with owner-only permissions.
    /// The start scripts pass `<NVMAI>/memory`, beside `models/`; the binary
    /// alone falls back to `~/.nvmai/memory`.
    public var directory: URL
    /// A project file untouched for this many days has its session log
    /// expired -- the transcript, which is the bulk of it -- and keeps its
    /// facts. Zero keeps everything. One file per project accumulates one
    /// per directory a client ever ran from, and nothing else would ever
    /// trim them; but a novel paused for six weeks must not come back
    /// without its bible, so retention never removes a fact. Only the
    /// project cap does.
    public var retentionDays: Int
    /// Most project files kept; the oldest by last write go first. Zero is
    /// no cap.
    public var maximumWorkspaces: Int
    /// Force every append to disk. Correct across a power cut and slower;
    /// a process crash loses nothing either way, because the write has
    /// already reached the kernel.
    public var synchronizesEveryWrite: Bool
    /// Optional ceiling for what the store may hold, in bytes, across every
    /// open workspace. Nil, the default, is no ceiling.
    ///
    /// There is no default because there is nothing to defend against.
    /// Measured on a hundred-chapter novel written over ten sessions, the
    /// whole store -- facts, their history and the session log -- was about
    /// 100 KB resident, and a three-language port was 10 KB. A ceiling sized
    /// for the machine was a rounding error next to one KV-cache block, and
    /// a bound that can never bind is a knob that only confuses. The
    /// hygiene bounds that are not budgets stay: a fact is at most 64 KiB, an
    /// address keeps 32 versions, the journal keeps 200 turns a session.
    ///
    /// `NVMAI_MEMORY_CACHE_MIB` sets one for anyone who wants it; at the cap
    /// facts refuse and the journal evicts, as before.
    public var maximumMemoryBytes: Int?

    public init(directory: URL = ContinuityStorageConfiguration.defaultDirectory,
                synchronizesEveryWrite: Bool = false,
                maximumMemoryBytes: Int? = nil,
                retentionDays: Int = 30,
                maximumWorkspaces: Int = 100) {
        self.directory = directory
        self.synchronizesEveryWrite = synchronizesEveryWrite
        self.maximumMemoryBytes = maximumMemoryBytes
        self.retentionDays = max(0, retentionDays)
        self.maximumWorkspaces = max(0, maximumWorkspaces)
    }

    public static var defaultDirectory: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".nvmai", isDirectory: true)
            .appendingPathComponent("memory", isDirectory: true)
    }

    /// How a ceiling, when one is set, is divided between the two stores.
    /// Zero for both means unbounded, which is the default.
    ///
    /// Facts get three quarters, the journal one. Facts are the half that has
    /// to be resident: they are what a session searches and what goes into a
    /// prompt. The journal never enters a prompt and every byte of it is
    /// already in the file, so its quarter is a window over recent sessions,
    /// not a home for them.
    public var budget: (factBytes: Int, logBytes: Int) {
        guard let total = maximumMemoryBytes, total > 0 else {
            return (factBytes: 0, logBytes: 0)
        }
        let log = max(1 << 20, total / 4)
        return (factBytes: max(1 << 20, total - log), logBytes: log)
    }

    /// The journal file for a scope. One file per workspace, so deleting a
    /// project's memory is deleting a file rather than editing a shared one.
    public func journalURL(for scope: MemoryScope) -> URL {
        directory
            .appendingPathComponent(scope.namespace, isDirectory: true)
            .appendingPathComponent(scope.user, isDirectory: true)
            .appendingPathComponent("\(scope.workspace).ndjson")
    }
}

/// The memory subsystem's whole configuration surface.
public struct MemoryConfiguration: Sendable, Equatable {
    public var isEnabled: Bool
    /// Why memory was asked for and not enabled, when that happened. Nil
    /// otherwise. The factory logs it, so a refusal is visible at start
    /// rather than discovered as facts from two projects in one bootstrap.
    public var disabledReason: String?
    public var storage: ContinuityStorageConfiguration
    /// Separates deployments sharing one machine.
    public var namespace: String
    /// Separates people sharing one server. Defaults to the OS user.
    public var user: String
    /// The repository or project this server is serving. Set at launch by the
    /// start scripts, overridable per request.
    public var workspace: String
    /// Whether a request may name its own workspace. On by default so one
    /// server can serve several checkouts; off pins the server to one.
    public var allowsPerRequestWorkspace: Bool
    public var limits: MemoryLimits
    /// Keys scanned from a scope's index before ranking. Bounds the cost of
    /// a search on a large store; nothing ever reads the whole database.
    public var maximumIndexScan: Int
    /// How much of the memory API the model is shown.
    ///
    /// Off by default. The tool loop is where the request-lifecycle risk and
    /// the dependence on the model's tool discipline concentrate, and whether
    /// a 3B-active model uses six memory tools well is a measurement rather
    /// than a claim. Bootstrap injection and the session journal carry the
    /// feature's value without it.
    public var toolSurface: MemoryToolSurface
    /// Rounds of memory tool calls the engine will service inside one
    /// request before it stops and answers.
    public var maximumToolRounds: Int
    /// Whether the engine writes a session journal. On when memory is on:
    /// it costs no tokens and no latency on the critical path, and it is the
    /// component whose value does not depend on the model's tool discipline.
    public var journalEnabled: Bool
    /// The journal's own limits, budgeted apart from curated memory.
    public var journalLimits: JournalLimits
    /// Ask the model, at session end, what is worth keeping. Off by default:
    /// it costs a generation the user did not ask for.
    public var sessionConsolidation: Bool
    /// Seconds of quiet after a turn before the session is consolidated.
    ///
    /// Consolidation is a full generation, and this machine is single-tenant,
    /// so it runs in the pauses -- while the person reads the reply or has
    /// walked away -- never in the request path. Thirty seconds is longer
    /// than reading a reply and shorter than starting the next conversation,
    /// so a session begun right after the last one usually finds it already
    /// distilled rather than one session stale. A turn arriving during one
    /// waits behind it, and only behind it.
    public var consolidationIdleSeconds: Double
    /// Most recent turns a consolidation reads. Bounds its prompt.
    public var consolidationMaximumTurns: Int
    /// Sessions whose whole transcript is shorter than this are not
    /// consolidated. A "say OK" probe or a one-line question has nothing
    /// durable in it, and measured, distilling one still cost eleven seconds
    /// of a 35B model to produce "[]". Set low: a person saying "remember
    /// that the town is Ashgrove" and a one-line confirmation is about two
    /// hundred characters and is exactly what must be kept.
    public var consolidationMinimumCharacters: Int
    /// Serve memory from process-local storage when the journal cannot be written,
    /// so a session still has working memory. It does not survive restart,
    /// and the model is told which one it is talking to.
    public var degradesToLocalStore: Bool

    public init(isEnabled: Bool = false,
                storage: ContinuityStorageConfiguration = .init(),
                namespace: String = "nvmai",
                user: String = MemoryConfiguration.defaultUser,
                workspace: String = "default",
                allowsPerRequestWorkspace: Bool = true,
                limits: MemoryLimits = .init(),
                maximumIndexScan: Int = 2_000,
                toolSurface: MemoryToolSurface = .off,
                maximumToolRounds: Int = 4,
                journalEnabled: Bool = true,
                journalLimits: JournalLimits = .init(),
                sessionConsolidation: Bool = true,
                consolidationIdleSeconds: Double = 30,
                consolidationMaximumTurns: Int = 40,
                consolidationMinimumCharacters: Int = 150,
                degradesToLocalStore: Bool = true) {
        self.isEnabled = isEnabled
        self.storage = storage
        self.namespace = namespace
        self.user = user
        self.workspace = workspace
        self.allowsPerRequestWorkspace = allowsPerRequestWorkspace
        self.limits = limits
        self.maximumIndexScan = maximumIndexScan
        self.toolSurface = toolSurface
        self.maximumToolRounds = maximumToolRounds
        self.journalEnabled = journalEnabled
        self.journalLimits = journalLimits
        self.sessionConsolidation = sessionConsolidation
        self.consolidationIdleSeconds = max(0, consolidationIdleSeconds)
        self.consolidationMaximumTurns = max(1, consolidationMaximumTurns)
        self.consolidationMinimumCharacters = max(0, consolidationMinimumCharacters)
        self.degradesToLocalStore = degradesToLocalStore
    }

    public static var defaultUser: String {
        let name = ProcessInfo.processInfo.environment["USER"] ?? NSUserName()
        let sanitized = name.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
        return sanitized.isEmpty ? "local" : String(sanitized.prefix(32))
    }

    /// Reads the configuration from the environment, which is how the start
    /// scripts and the launchers pass it.
    ///
    /// Every value has a default that works on a developer machine with a
    /// machine, and the subsystem stays off unless NVMAI_MEMORY is set,
    /// so nothing about serving changes for someone who has not asked for it.
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        let flag = environment["NVMAI_MEMORY"]?.lowercased()
        configuration.isEnabled = flag == "1" || flag == "on" || flag == "true"

        if let directory = environment["NVMAI_MEMORY_DIR"], !directory.isEmpty {
            configuration.storage.directory = URL(fileURLWithPath: directory)
        }
        if let value = environment["NVMAI_MEMORY_FSYNC"] {
            configuration.storage.synchronizesEveryWrite = value == "1"
        }
        if let value = environment["NVMAI_MEMORY_RETENTION_DAYS"].flatMap(Int.init) {
            configuration.storage.retentionDays = max(0, value)
        }
        if let value = environment["NVMAI_MEMORY_MAX_WORKSPACES"].flatMap(Int.init) {
            configuration.storage.maximumWorkspaces = max(0, value)
        }
        if let cacheMiB = environment["NVMAI_MEMORY_CACHE_MIB"].flatMap(Int.init), cacheMiB > 0 {
            configuration.storage.maximumMemoryBytes = cacheMiB << 20
        }

        if let namespace = environment["NVMAI_MEMORY_NAMESPACE"] { configuration.namespace = namespace }
        if let user = environment["NVMAI_MEMORY_USER"] { configuration.user = user }
        if let workspace = environment["NVMAI_MEMORY_WORKSPACE"] {
            configuration.workspace = workspace
        } else if let directory = environment["NVMAI_WORKSPACE_DIR"] {
            // Only a refusal when memory was actually asked for. Off is off,
            // and a reason that begins "NVMAI_MEMORY=1 but" must never be
            // logged for someone who never set it.
            if configuration.isEnabled,
               let reason = junkDrawerReason(forPath: directory, environment: environment) {
                // A server launched from the home directory and used for
                // everything would put a novel and a codebase in one fact
                // store. Refusing is the only outcome that is visible.
                configuration.isEnabled = false
                configuration.disabledReason = reason
            } else {
                configuration.workspace = workspaceIdentifier(forPath: directory)
            }
        }
        if let value = environment["NVMAI_MEMORY_MAX_VALUE_BYTES"].flatMap(Int.init) {
            configuration.limits.maximumValueBytes = max(256, value)
        }
        if let value = environment["NVMAI_MEMORY_BOOTSTRAP_LIMIT"].flatMap(Int.init) {
            configuration.limits.bootstrapRecords = max(0, value)
        }
        if let value = environment["NVMAI_MEMORY_BOOTSTRAP_BYTES"].flatMap(Int.init) {
            configuration.limits.bootstrapBytes = max(0, value)
        }
        if let value = environment["NVMAI_MEMORY_TOOL_ROUNDS"].flatMap(Int.init) {
            configuration.maximumToolRounds = max(0, min(value, 16))
        }
        if let value = environment["NVMAI_MEMORY_TOOLS"] {
            // "1" and "0" kept working: they predate the surface.
            switch value.lowercased() {
            case "1", "full", "on": configuration.toolSurface = .full
            case "0", "off": configuration.toolSurface = .off
            default: configuration.toolSurface = MemoryToolSurface(rawValue: value.lowercased())
                ?? configuration.toolSurface
            }
        }
        if let value = environment["NVMAI_MEMORY_JOURNAL"] {
            configuration.journalEnabled = value != "0"
        }
        if let value = environment["NVMAI_MEMORY_JOURNAL_TURNS"].flatMap(Int.init) {
            configuration.journalLimits.turnsPerSession = max(1, value)
        }
        if let value = environment["NVMAI_MEMORY_JOURNAL_SESSIONS"].flatMap(Int.init) {
            configuration.journalLimits.sessionsPerWorkspace = max(1, value)
        }
        if let value = environment["NVMAI_MEMORY_CONSOLIDATION"] {
            configuration.sessionConsolidation = value != "0"
        }
        if let value = environment["NVMAI_MEMORY_CONSOLIDATION_IDLE_SECONDS"].flatMap(Double.init) {
            configuration.consolidationIdleSeconds = max(0, value)
        }
        if let value = environment["NVMAI_MEMORY_LOCAL_FALLBACK"] {
            configuration.degradesToLocalStore = value != "0"
        }
        return configuration
    }

    /// Why a launch directory cannot be a workspace, or nil when it can.
    ///
    /// The home directory, its parent and the root are not projects; they
    /// are where someone happens to have a terminal open. A workspace named
    /// after one of them collects every project that person ever works on,
    /// and the bootstrap for a codebase then opens with the plot of a novel.
    public static func junkDrawerReason(forPath path: String,
                                        environment: [String: String]) -> String? {
        let candidate = URL(fileURLWithPath: path).standardizedFileURL.path
        let home = (environment["HOME"]
                    ?? FileManager.default.homeDirectoryForCurrentUser.path)
        let homePath = URL(fileURLWithPath: home).standardizedFileURL.path
        let refused: [(String, String)] = [
            (homePath, "the home directory"),
            (URL(fileURLWithPath: homePath).deletingLastPathComponent().path,
             "the parent of the home directory"),
            ("/", "the filesystem root"),
        ]
        for (refusedPath, label) in refused where candidate == refusedPath {
            return "NVMAI_MEMORY=1 but the launch directory is \(label) (\(candidate)), "
                + "which is not a project; memory stays off. Launch from the project "
                + "directory, or set NVMAI_MEMORY_WORKSPACE=<name>."
        }
        return nil
    }

    /// A stable workspace id from a filesystem path: the directory name, plus
    /// a short digest of the full path so two checkouts of the same
    /// repository do not share memory.
    public static func workspaceIdentifier(forPath path: String) -> String {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let name = url.lastPathComponent.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
        let digest = String(format: "%08x", UInt32(truncatingIfNeeded: stableHash(url.path)))
        let base = name.isEmpty ? "workspace" : String(name.prefix(40))
        return "\(base)-\(digest)"
    }

    /// FNV-1a. Swift's `hashValue` is seeded per process, so it cannot name a
    /// workspace that has to be the same after a restart.
    private static func stableHash(_ text: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    /// The workspace that holds the person's own facts -- conventions,
    /// language, tone -- shown in every project. Reserved: a request may not
    /// name it, and a derived workspace id can never spell it, so nothing a
    /// client sends can write project state into it.
    public static let sharedWorkspace = "_global"

    /// The scope of the person's shared facts.
    public var sharedScope: MemoryScope? {
        try? MemoryScope(namespace: namespace, user: user, workspace: Self.sharedWorkspace)
    }

    /// The scope this configuration describes, or nil when a component is
    /// unusable. A rejected scope disables memory rather than falling back to
    /// a shared one, because the failure mode of guessing is cross-project
    /// leakage.
    public func scope(workspaceOverride: String? = nil) -> MemoryScope? {
        let effective = allowsPerRequestWorkspace ? (workspaceOverride ?? workspace) : workspace
        guard effective != Self.sharedWorkspace else { return nil }
        return try? MemoryScope(namespace: namespace, user: user, workspace: effective)
    }

    /// One line for the log at startup.
    public var summary: String {
        let cache = storage.maximumMemoryBytes.map { "cap=\($0 >> 20)MiB" } ?? "cap=none"
        return "memory enabled=\(isEnabled) store=in-process \(cache) "
            + "namespace=\(namespace) user=\(user) workspace=\(workspace) "
            + "memory_tools=\(toolSurface.rawValue) memory_tool_rounds=\(maximumToolRounds) "
            + "bootstrap=\(limits.bootstrapRecords)/\(limits.bootstrapBytes)B "
            + "dir=\(storage.directory.path) retention=\(storage.retentionDays)d "
            + "max_workspaces=\(storage.maximumWorkspaces) "
            + "journal=\(journalEnabled) "
            + "journal_limits=\(journalLimits.turnsPerSession)/"
            + "\(journalLimits.sessionsPerWorkspace) "
            + "consolidation=\(sessionConsolidation) idle=\(Int(consolidationIdleSeconds))s"
    }
}
