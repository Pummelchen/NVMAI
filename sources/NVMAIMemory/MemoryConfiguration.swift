import Foundation

/// Where continuity state lives on this machine.
///
/// There is no host and no port. The store runs inside the server process,
/// so the only things left to configure are the directory it writes to and
/// how much it is allowed to hold.
public struct ContinuityStorageConfiguration: Sendable, Equatable {
    /// Directory holding the journal. Created with owner-only permissions.
    public var directory: URL
    /// Force every append to disk. Correct across a power cut and slower;
    /// a process crash loses nothing either way, because the write has
    /// already reached the kernel.
    public var synchronizesEveryWrite: Bool
    /// Ceiling for what the store may hold, in bytes. Nil leaves it unbounded,
    /// which is only sensible in tests.
    public var maximumMemoryBytes: Int?

    public init(directory: URL = ContinuityStorageConfiguration.defaultDirectory,
                synchronizesEveryWrite: Bool = false,
                maximumMemoryBytes: Int? = nil) {
        self.directory = directory
        self.synchronizesEveryWrite = synchronizesEveryWrite
        self.maximumMemoryBytes = maximumMemoryBytes
    }

    public static var defaultDirectory: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".nvmai", isDirectory: true)
            .appendingPathComponent("memory", isDirectory: true)
    }

    /// How the ceiling is divided between the two stores.
    ///
    /// Facts and turns compete for one budget and are not comparable: a fact
    /// is a sentence that took a model deliberate effort to write, a turn is
    /// kilobytes of prose the engine captured for free. Turns are also the
    /// side that grows without limit. So the split is deliberately lopsided
    /// and the facts' share is the protected one.
    public var budget: (factBytes: Int, logBytes: Int) {
        guard let total = maximumMemoryBytes, total > 0 else {
            return (factBytes: 64 << 20, logBytes: 192 << 20)
        }
        let facts = max(1 << 20, total / 4)
        return (factBytes: facts, logBytes: max(1 << 20, total - facts))
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
                sessionConsolidation: Bool = false,
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
        self.degradesToLocalStore = degradesToLocalStore
    }

    public static var defaultUser: String {
        let name = ProcessInfo.processInfo.environment["USER"] ?? NSUserName()
        let sanitized = name.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
        return sanitized.isEmpty ? "local" : String(sanitized.prefix(32))
    }

    /// Default store ceiling for this machine, following the sizing the
    /// deployment asks for: 256 MiB at 8 GB, 512 MiB at 16 GB, 1 GiB above.
    /// A memory store is worth a fixed slice, not a fraction: the working set
    /// is a few thousand short facts and does not grow with the machine.
    public static func defaultCacheBytes(
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let gigabyte = UInt64(1) << 30
        if physicalMemory <= 8 * gigabyte { return 256 << 20 }
        if physicalMemory <= 16 * gigabyte { return 512 << 20 }
        return 1 << 30
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
        let cacheMiB = environment["NVMAI_MEMORY_CACHE_MIB"].flatMap(Int.init)
        configuration.storage.maximumMemoryBytes = cacheMiB.map { $0 << 20 } ?? defaultCacheBytes()

        if let namespace = environment["NVMAI_MEMORY_NAMESPACE"] { configuration.namespace = namespace }
        if let user = environment["NVMAI_MEMORY_USER"] { configuration.user = user }
        if let workspace = environment["NVMAI_MEMORY_WORKSPACE"] {
            configuration.workspace = workspace
        } else if let directory = environment["NVMAI_WORKSPACE_DIR"] {
            configuration.workspace = workspaceIdentifier(forPath: directory)
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
            configuration.sessionConsolidation = value == "1"
        }
        if let value = environment["NVMAI_MEMORY_LOCAL_FALLBACK"] {
            configuration.degradesToLocalStore = value != "0"
        }
        return configuration
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

    /// The scope this configuration describes, or nil when a component is
    /// unusable. A rejected scope disables memory rather than falling back to
    /// a shared one, because the failure mode of guessing is cross-project
    /// leakage.
    public func scope(workspaceOverride: String? = nil) -> MemoryScope? {
        let effective = allowsPerRequestWorkspace ? (workspaceOverride ?? workspace) : workspace
        return try? MemoryScope(namespace: namespace, user: user, workspace: effective)
    }

    /// One line for the log at startup.
    public var summary: String {
        let cache = storage.maximumMemoryBytes.map { "\($0 >> 20)MiB" } ?? "unbounded"
        return "memory enabled=\(isEnabled) store=in-process cache=\(cache) "
            + "namespace=\(namespace) user=\(user) workspace=\(workspace) "
            + "tools=\(toolSurface.rawValue) rounds=\(maximumToolRounds) "
            + "bootstrap=\(limits.bootstrapRecords)/\(limits.bootstrapBytes)B "
            + "dir=\(storage.directory.path) "
            + "journal=\(journalEnabled) "
            + "journal_limits=\(journalLimits.turnsPerSession)/"
            + "\(journalLimits.sessionsPerWorkspace) "
            + "consolidation=\(sessionConsolidation)"
    }
}
