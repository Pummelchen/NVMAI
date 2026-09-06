import Foundation
import ContinuityCore

/// A project file read without opening it for writing.
///
/// This is what a person uses the day memory says something wrong: list the
/// facts, read one, see how it changed. It takes no lock, so it works on the
/// file of a server that is running, and it folds the journal the same way
/// the engine does on start -- a checkpoint resets, a memory record upserts,
/// a version extends the history.
public struct MemoryProjectFile: Sendable {
    public struct Fact: Sendable, Equatable {
        public let key: String
        public let value: String
        public let version: Int
        public let status: String
        public let updatedAt: Date
        /// Earlier values, oldest first.
        public let history: [(version: Int, value: String)]

        public static func == (lhs: Fact, rhs: Fact) -> Bool {
            lhs.key == rhs.key && lhs.version == rhs.version && lhs.value == rhs.value
        }
    }

    public let url: URL
    /// The workspace id, which is the file's stem.
    public let workspace: String
    public let title: String?
    public let facts: [Fact]
    public let sessionCount: Int
    public let eventCount: Int
    public let bytesOnDisk: Int
    public let modifiedAt: Date

    /// Every project file under a memory directory, newest first.
    public static func discover(in directory: URL) -> [MemoryProjectFile] {
        let manager = FileManager.default
        guard let walker = manager.enumerator(at: directory,
                                              includingPropertiesForKeys: [.contentModificationDateKey],
                                              options: [.skipsHiddenFiles]) else { return [] }
        var files: [MemoryProjectFile] = []
        for case let url as URL in walker where url.pathExtension == "ndjson" {
            if let file = try? load(url) { files.append(file) }
        }
        return files.sorted { $0.modifiedAt > $1.modifiedAt }
    }

    /// Picks a project by exact workspace id or by a unique prefix, so
    /// `photograph` finds `photograph-851a1c1a`.
    public static func resolve(_ name: String, among files: [MemoryProjectFile])
        -> Result<MemoryProjectFile, ResolveError> {
        if let exact = files.first(where: { $0.workspace == name }) { return .success(exact) }
        let matches = files.filter { $0.workspace.hasPrefix(name) }
        switch matches.count {
        case 0: return .failure(.notFound(name))
        case 1: return .success(matches[0])
        default: return .failure(.ambiguous(name, matches.map(\.workspace)))
        }
    }

    public enum ResolveError: Error, CustomStringConvertible {
        case notFound(String)
        case ambiguous(String, [String])
        public var description: String {
            switch self {
            case .notFound(let name): return "no project matches '\(name)'"
            case .ambiguous(let name, let candidates):
                return "'\(name)' matches several projects: " + candidates.joined(separator: ", ")
            }
        }
    }

    public static func load(_ url: URL) throws -> MemoryProjectFile {
        let records = try FileJournal.read(contentsOf: url)
        var items: [String: MemoryItem] = [:]
        var versions: [String: [(Int, String)]] = [:]
        var tasks: [UUID: ContinuityTask] = [:]
        var sessions = Set<UUID>()
        var events = 0
        for record in records {
            switch record {
            case .checkpoint(let log, let memory):
                items = [:]; versions = [:]; tasks = [:]; sessions = []
                for task in log.tasks { tasks[task.id] = task }
                for session in log.sessions { sessions.insert(session.id) }
                events = log.events.count
                for item in memory.items { items[item.address] = item }
                for version in memory.versions {
                    versions["\(version.namespace).\(version.key)", default: []]
                        .append((version.version, version.value))
                }
            case .task(let task): tasks[task.id] = task
            case .session(let session): sessions.insert(session.id)
            case .event: events += 1
            case .memory(let item): items[item.address] = item
            case .memoryVersion(let version):
                versions["\(version.namespace).\(version.key)", default: []]
                    .append((version.version, version.value))
            }
        }
        let facts = items.values.map { item in
            Fact(key: ContinuityStore.keyText(for: item),
                 value: item.value,
                 version: item.version,
                 status: item.status.rawValue,
                 updatedAt: item.updatedAt,
                 history: (versions[item.address] ?? []).sorted { $0.0 < $1.0 }
                     .map { (version: $0.0, value: $0.1) })
        }.sorted { $0.key < $1.key }
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return MemoryProjectFile(
            url: url,
            workspace: url.deletingPathExtension().lastPathComponent,
            title: tasks.values.first?.title,
            facts: facts,
            sessionCount: sessions.count,
            eventCount: events,
            bytesOnDisk: (attributes?[.size] as? Int) ?? 0,
            modifiedAt: (attributes?[.modificationDate] as? Date) ?? .distantPast)
    }

    /// Where the project files are, by the same rule the server uses:
    /// `NVMAI_MEMORY_DIR`, else `memory/` under the current directory when it
    /// exists, else the binary's own fallback.
    public static func defaultDirectory(environment: [String: String] = ProcessInfo.processInfo.environment,
                                        currentDirectory: String = FileManager.default.currentDirectoryPath)
        -> URL {
        if let explicit = environment["NVMAI_MEMORY_DIR"], !explicit.isEmpty {
            return URL(fileURLWithPath: explicit)
        }
        let local = URL(fileURLWithPath: currentDirectory).appendingPathComponent("memory")
        if FileManager.default.fileExists(atPath: local.path) { return local }
        return ContinuityStorageConfiguration.defaultDirectory
    }
}
