import Foundation

/// How to reach Valkey. Parsed from a URL plus explicit overrides, so
/// `redis://user:pass@host:6379/2` works and so does setting the pieces one
/// at a time.
public struct ValkeyConfiguration: Sendable, Equatable {
    public var host: String
    public var port: Int
    public var username: String?
    public var password: String?
    public var database: Int
    public var connectTimeoutMilliseconds: Int
    public var operationTimeoutMilliseconds: Int
    /// Applied with `CONFIG SET maxmemory` at connect. Nil leaves the
    /// server's own configuration alone.
    public var maximumMemoryBytes: Int?

    public init(host: String = "127.0.0.1",
                port: Int = 6379,
                username: String? = nil,
                password: String? = nil,
                database: Int = 0,
                connectTimeoutMilliseconds: Int = 1_000,
                operationTimeoutMilliseconds: Int = 250,
                maximumMemoryBytes: Int? = nil) {
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.connectTimeoutMilliseconds = connectTimeoutMilliseconds
        self.operationTimeoutMilliseconds = operationTimeoutMilliseconds
        self.maximumMemoryBytes = maximumMemoryBytes
    }

    /// Parses `redis://`, `rediss://` or `valkey://` URLs. Credentials in the
    /// URL are supported because that is how deployments pass them; they are
    /// never logged and never reach the model.
    public static func parse(url raw: String) -> ValkeyConfiguration? {
        guard let url = URL(string: raw),
              let scheme = url.scheme?.lowercased(),
              ["redis", "rediss", "valkey"].contains(scheme) else { return nil }
        var configuration = ValkeyConfiguration()
        configuration.host = url.host ?? "127.0.0.1"
        configuration.port = url.port ?? 6379
        if let user = url.user, !user.isEmpty { configuration.username = user }
        if let password = url.password, !password.isEmpty { configuration.password = password }
        let path = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if let database = Int(path) { configuration.database = database }
        return configuration
    }
}

/// The memory subsystem's whole configuration surface.
public struct MemoryConfiguration: Sendable, Equatable {
    public var isEnabled: Bool
    public var valkey: ValkeyConfiguration
    /// Separates deployments sharing one Valkey.
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
    /// Whether the engine offers the model memory tools at all.
    public var exposesTools: Bool
    /// Rounds of memory tool calls the engine will service inside one
    /// request before it stops and answers.
    public var maximumToolRounds: Int
    /// Ask the model, at session end, what is worth keeping. Off by default:
    /// it costs a generation the user did not ask for.
    public var sessionConsolidation: Bool
    /// Serve memory from process-local storage when Valkey cannot be reached,
    /// so a session still has working memory. It does not survive restart,
    /// and the model is told which one it is talking to.
    public var degradesToLocalStore: Bool

    public init(isEnabled: Bool = false,
                valkey: ValkeyConfiguration = .init(),
                namespace: String = "nvmai",
                user: String = MemoryConfiguration.defaultUser,
                workspace: String = "default",
                allowsPerRequestWorkspace: Bool = true,
                limits: MemoryLimits = .init(),
                maximumIndexScan: Int = 2_000,
                exposesTools: Bool = true,
                maximumToolRounds: Int = 4,
                sessionConsolidation: Bool = false,
                degradesToLocalStore: Bool = true) {
        self.isEnabled = isEnabled
        self.valkey = valkey
        self.namespace = namespace
        self.user = user
        self.workspace = workspace
        self.allowsPerRequestWorkspace = allowsPerRequestWorkspace
        self.limits = limits
        self.maximumIndexScan = maximumIndexScan
        self.exposesTools = exposesTools
        self.maximumToolRounds = maximumToolRounds
        self.sessionConsolidation = sessionConsolidation
        self.degradesToLocalStore = degradesToLocalStore
    }

    public static var defaultUser: String {
        let name = ProcessInfo.processInfo.environment["USER"] ?? NSUserName()
        let sanitized = name.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
        return sanitized.isEmpty ? "local" : String(sanitized.prefix(32))
    }

    /// Default Valkey ceiling for this machine, following the sizing the
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
    /// local Valkey, and the subsystem stays off unless NVMAI_MEMORY is set,
    /// so nothing about serving changes for someone who has not asked for it.
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        let flag = environment["NVMAI_MEMORY"]?.lowercased()
        configuration.isEnabled = flag == "1" || flag == "on" || flag == "true"

        if let url = environment["VALKEY_URL"] ?? environment["NVMAI_MEMORY_URL"],
           let parsed = ValkeyConfiguration.parse(url: url) {
            configuration.valkey = parsed
        }
        if let host = environment["VALKEY_HOST"] { configuration.valkey.host = host }
        if let port = environment["VALKEY_PORT"].flatMap(Int.init) { configuration.valkey.port = port }
        if let user = environment["VALKEY_USERNAME"] { configuration.valkey.username = user }
        if let password = environment["VALKEY_PASSWORD"] { configuration.valkey.password = password }
        if let database = environment["VALKEY_DB"].flatMap(Int.init) {
            configuration.valkey.database = database
        }
        if let value = environment["NVMAI_MEMORY_CONNECT_TIMEOUT_MS"].flatMap(Int.init) {
            configuration.valkey.connectTimeoutMilliseconds = max(10, value)
        }
        if let value = environment["NVMAI_MEMORY_TIMEOUT_MS"].flatMap(Int.init) {
            configuration.valkey.operationTimeoutMilliseconds = max(10, value)
        }
        let cacheMiB = environment["NVMAI_MEMORY_CACHE_MIB"].flatMap(Int.init)
        configuration.valkey.maximumMemoryBytes = cacheMiB.map { $0 << 20 } ?? defaultCacheBytes()

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
        if let value = environment["NVMAI_MEMORY_TOOLS"] { configuration.exposesTools = value != "0" }
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

    /// One line for the log at startup. Never includes credentials.
    public var summary: String {
        let target = "\(valkey.host):\(valkey.port)/\(valkey.database)"
        let cache = valkey.maximumMemoryBytes.map { "\($0 >> 20)MiB" } ?? "server default"
        return "memory enabled=\(isEnabled) valkey=\(target) cache=\(cache) "
            + "namespace=\(namespace) user=\(user) workspace=\(workspace) "
            + "tools=\(exposesTools) rounds=\(maximumToolRounds) "
            + "bootstrap=\(limits.bootstrapRecords)/\(limits.bootstrapBytes)B "
            + "timeout=\(valkey.operationTimeoutMilliseconds)ms "
            + "consolidation=\(sessionConsolidation)"
    }
}
