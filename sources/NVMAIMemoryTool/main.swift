import Foundation
import ContinuityCore
import NVMAIMemory

/// `nvmai-memory`: see and correct what the server remembers.
///
///   nvmai-memory projects                  every project file, newest first
///   nvmai-memory list <project>            the facts a project holds
///   nvmai-memory show <project> <key>      one fact, in full, with its history
///   nvmai-memory delete <project> <key>    retire a fact (kept in history, never shown again)
///   nvmai-memory forget <project> --yes    delete a project's memory outright
///
/// `<project>` is a workspace id or a unique prefix of one; `global` is the
/// person's shared facts. `--dir` overrides NVMAI_MEMORY_DIR. Reads take no
/// lock and work while a server is running; delete and forget need the
/// workspace, and refuse it while a server holds it.

let usage = """
usage: nvmai-memory [--dir DIR] projects
       nvmai-memory [--dir DIR] list <project>
       nvmai-memory [--dir DIR] show <project> <key>
       nvmai-memory [--dir DIR] delete <project> <key>
       nvmai-memory [--dir DIR] forget <project> --yes
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("nvmai-memory: \(message)\n".utf8))
    exit(1)
}

var arguments = Array(CommandLine.arguments.dropFirst())
var directory = MemoryProjectFile.defaultDirectory()
if let index = arguments.firstIndex(of: "--dir") {
    guard index + 1 < arguments.count else { fail("--dir needs a path") }
    directory = URL(fileURLWithPath: arguments[index + 1])
    arguments.removeSubrange(index...index + 1)
}
let yes = arguments.contains("--yes")
arguments.removeAll { $0 == "--yes" }
guard let command = arguments.first else { print(usage); exit(2) }

func projectName(_ raw: String) -> String {
    raw == "global" ? MemoryConfiguration.sharedWorkspace : raw
}

func resolve(_ raw: String, in directory: URL) -> MemoryProjectFile {
    let files = MemoryProjectFile.discover(in: directory)
    switch MemoryProjectFile.resolve(projectName(raw), among: files) {
    case .success(let file): return file
    case .failure(let error): fail(error.description)
    }
}

func summarize(_ value: String, limit: Int = 100) -> String {
    let flat = value.replacingOccurrences(of: "\n", with: " ")
    return flat.count > limit ? String(flat.prefix(limit)) + "…" : flat
}

let formatter = DateFormatter()
formatter.dateFormat = "yyyy-MM-dd HH:mm"

switch command {
case "projects":
    let files = MemoryProjectFile.discover(in: directory)
    guard !files.isEmpty else { print("no project files under \(directory.path)"); exit(0) }
    print("\(directory.path)\n")
    print(String(format: "%-40s %6s %8s %8s %10s  %s", "project", "facts", "sessions", "events", "on disk", "last write"))
    for file in files {
        let label = file.workspace == MemoryConfiguration.sharedWorkspace ? "global (this person)" : file.workspace
        print(String(format: "%-40s %6d %8d %8d %9dK  %s", label, file.facts.count, file.sessionCount,
                     file.eventCount, file.bytesOnDisk / 1024, formatter.string(from: file.modifiedAt)))
    }

case "list":
    guard arguments.count >= 2 else { fail("list needs a project") }
    let file = resolve(arguments[1], in: directory)
    print("\(file.workspace)\(file.title.map { " — \($0)" } ?? "")  (\(file.facts.count) facts)\n")
    for fact in file.facts where fact.status != "archived" {
        let marker = fact.status == "disputed" ? " [disputed]" : ""
        print("  \(fact.key) (v\(fact.version))\(marker): \(summarize(fact.value))")
    }
    let archived = file.facts.filter { $0.status == "archived" }.count
    if archived > 0 { print("\n  \(archived) archived fact(s) not shown") }

case "show":
    guard arguments.count >= 3 else { fail("show needs a project and a key") }
    let file = resolve(arguments[1], in: directory)
    guard let fact = file.facts.first(where: { $0.key == arguments[2] }) else {
        fail("no fact '\(arguments[2])' in \(file.workspace)")
    }
    print("\(fact.key)  [\(fact.status), v\(fact.version), \(formatter.string(from: fact.updatedAt))]\n")
    print(fact.value)
    if !fact.history.isEmpty {
        print("\nearlier:")
        for entry in fact.history { print("  v\(entry.version): \(summarize(entry.value, limit: 160))") }
    }

case "delete", "forget":
    guard arguments.count >= 2 else { fail("\(command) needs a project") }
    let file = resolve(arguments[1], in: directory)
    if command == "forget" {
        guard yes else { fail("this deletes every fact and session of \(file.workspace); repeat with --yes") }
        try? FileManager.default.removeItem(at: file.url.appendingPathExtension("lock"))
        do { try FileManager.default.removeItem(at: file.url) } catch { fail("could not delete: \(error)") }
        print("forgot \(file.workspace)")
        exit(0)
    }
    guard arguments.count >= 3 else { fail("delete needs a project and a key") }
    let key: MemoryKey
    do { key = try MemoryKey(validating: arguments[2]) } catch { fail("\(error)") }
    let journal: FileJournal
    do { journal = try FileJournal(url: file.url) } catch {
        fail("\(error)\nStop the server that has this project open, or delete through it.")
    }
    let engine = ContinuityEngine(journal: journal)
    do {
        try await engine.start()
        let address = ContinuityStore.address(for: key)
        guard let task = await engine.tasks().first else { fail("no task in this file") }
        _ = try await engine.archive(taskID: task.id, namespace: address.namespace, key: address.key)
        try await engine.compactJournal()
        await engine.shutDown()
    } catch {
        await engine.shutDown()
        fail("\(error)")
    }
    print("retired \(key.rawValue) in \(file.workspace); it stays in history and is no longer shown")

default:
    print(usage); exit(2)
}
