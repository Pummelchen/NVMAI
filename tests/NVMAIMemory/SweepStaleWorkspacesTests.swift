import Foundation
import Testing
@testable import NVMAIMemory

/// The workspace cap is the only retention rule that *removes facts*, so what it
/// is allowed to delete matters more than what it does delete.
@Suite struct SweepStaleWorkspacesTests {

    private func makeDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("nvmai-sweep-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ url: URL, modified: Date) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("{}\n".utf8).write(to: url)
        try FileManager.default.setAttributes([.modificationDate: modified],
                                              ofItemAtPath: url.path)
    }

    /// Cap only: retention rewrites survivors through the journal, which is a
    /// different path (and takes the lock itself).
    private func configuration(directory: URL, cap: Int) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.storage.directory = directory
        configuration.storage.maximumWorkspaces = cap
        configuration.storage.retentionDays = 0
        return configuration
    }

    /// Holds a workspace's lock the way `FileJournal` does. `flock` conflicts
    /// between open-file-descriptions even inside one process, so this stands in
    /// for another process exactly as far as the probe is concerned.
    private func lock(_ journalURL: URL) -> Int32 {
        let path = journalURL.appendingPathExtension("lock").path
        let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
        guard descriptor >= 0, flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            Issue.record("could not take the workspace lock at \(path)")
            if descriptor >= 0 { close(descriptor) }
            return -1
        }
        return descriptor
    }

    /// Three workspaces and a cap of one: the two oldest are doomed, and the
    /// oldest of all is held by "another process". Before the lock probe both
    /// were deleted, taking the holder's `.lock` with them.
    @Test func theCapLeavesALockedWorkspaceAlone() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let now = Date()
        let locked = directory.appendingPathComponent("nvmai/local/locked.ndjson")
        let free = directory.appendingPathComponent("nvmai/local/free.ndjson")
        let kept = directory.appendingPathComponent("nvmai/local/kept.ndjson")
        try write(locked, modified: now.addingTimeInterval(-300))
        try write(free, modified: now.addingTimeInterval(-200))
        try write(kept, modified: now.addingTimeInterval(-100))
        let descriptor = lock(locked)
        defer { if descriptor >= 0 { close(descriptor) } }

        let service = MemoryService(configuration: configuration(directory: directory, cap: 1))
        await service.sweepStaleWorkspaces()

        let manager = FileManager.default
        #expect(manager.fileExists(atPath: locked.path),
                "a workspace another process holds was deleted")
        #expect(manager.fileExists(atPath: locked.appendingPathExtension("lock").path),
                "the holder's lock file was deleted with it")
        #expect(manager.fileExists(atPath: kept.path),
                "the newest workspace is inside the cap and must survive")
        #expect(!manager.fileExists(atPath: free.path),
                "an unlocked workspace past the cap was not deleted")
    }

    /// The probe must not amount to disabling the cap: with no lock held, the
    /// same sweep deletes the same file.
    @Test func theCapStillDeletesAnUnlockedWorkspace() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let now = Date()
        let older = directory.appendingPathComponent("nvmai/local/older.ndjson")
        let newer = directory.appendingPathComponent("nvmai/local/newer.ndjson")
        try write(older, modified: now.addingTimeInterval(-300))
        try write(newer, modified: now.addingTimeInterval(-100))

        let service = MemoryService(configuration: configuration(directory: directory, cap: 1))
        await service.sweepStaleWorkspaces()

        let manager = FileManager.default
        #expect(!manager.fileExists(atPath: older.path))
        #expect(manager.fileExists(atPath: newer.path))
    }

    /// A workspace with no `.lock` file at all was never opened by a journal, so
    /// nothing can be holding it and the probe must say so rather than treating
    /// the missing file as "in use" (which would disable the cap for every
    /// workspace that has ever been idle).
    @Test func aWorkspaceWithoutALockFileIsNotHeld() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journal = directory.appendingPathComponent("nvmai/local/never-opened.ndjson")
        try write(journal, modified: Date())
        #expect(!MemoryService.isLockHeld(at: journal))
    }
}
