import Darwin
import Foundation

public struct RemoteInstallPaths: Sendable, Equatable {
    public let parentDirectory: String
    public let finalDirectory: String
    public let partialDirectory: String
    public let lockFile: String
    public let checkpointFile: String
    public let rangeTemporaryFile: String
    public let metadataDirectory: String

    public init(outputDirectory: String) throws {
        let standardized = URL(fileURLWithPath: outputDirectory).standardizedFileURL
        let basename = standardized.lastPathComponent.precomposedStringWithCanonicalMapping
        guard !basename.isEmpty,
              basename != ".",
              basename != "..",
              !basename.contains("/") else {
            throw RepackError.installPathUnsafe(
                path: outputDirectory,
                detail: "invalid output basename")
        }
        let requestedParent = standardized.deletingLastPathComponent().path
        try Posix.mkdirP(requestedParent)
        let parent = try Posix.physicalPath(requestedParent)
        let final = (parent as NSString).appendingPathComponent(basename)
        let partial = final + ".partial"
        self.parentDirectory = parent
        self.finalDirectory = final
        self.partialDirectory = partial
        self.lockFile = final + ".install.lock"
        self.checkpointFile = final + ".resume.json"
        self.rangeTemporaryFile = (partial as NSString)
            .appendingPathComponent(".range.tmp")
        self.metadataDirectory = (partial as NSString)
            .appendingPathComponent(".remote-metadata")
    }

    public func validateEntryTypes() throws {
        try require(finalDirectory, kinds: [.absent, .directory])
        try require(partialDirectory, kinds: [.absent, .directory])
        try require(checkpointFile, kinds: [.absent, .regular])
        try require(lockFile, kinds: [.absent, .regular])
    }

    private func require(_ path: String, kinds: Set<Posix.EntryKind>) throws {
        let kind = try Posix.entryKind(path)
        guard kinds.contains(kind) else {
            throw RepackError.installPathUnsafe(
                path: path,
                detail: "unexpected \(kind) entry")
        }
    }
}

public final class InstallLock: @unchecked Sendable {
    public let paths: RemoteInstallPaths
    private let descriptor: Int32

    private init(paths: RemoteInstallPaths, descriptor: Int32) {
        self.paths = paths
        self.descriptor = descriptor
    }

    deinit {
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
    }

    public static func acquire(outputDirectory: String) throws -> InstallLock {
        let paths = try RemoteInstallPaths(outputDirectory: outputDirectory)
        try paths.validateEntryTypes()
        let descriptor = try Posix.openLock(paths.lockFile)
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let savedErrno = errno
            close(descriptor)
            if savedErrno == EWOULDBLOCK {
                throw RepackError.installBusy(path: paths.lockFile)
            }
            throw RepackError.fileOpenFailed(path: paths.lockFile, errno: savedErrno)
        }
        do {
            try paths.validateEntryTypes()
            guard try Posix.descriptorMatchesPath(descriptor, path: paths.lockFile) else {
                throw RepackError.installPathUnsafe(
                    path: paths.lockFile,
                    detail: "lock path was replaced during acquisition")
            }
            // Close the validate-then-use TOCTOU window for the state paths:
            // re-open each present entry with O_NOFOLLOW and validate the
            // opened descriptor's type, instead of trusting the earlier lstat.
            try verifyOpenedEntry(path: paths.partialDirectory,
                                  allowedKinds: [.directory])
            try verifyOpenedEntry(path: paths.checkpointFile,
                                  allowedKinds: [.regular])
            return InstallLock(paths: paths, descriptor: descriptor)
        } catch {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
            throw error
        }
    }

    /// Opens `path` with O_NOFOLLOW and confirms the descriptor is one of the
    /// allowed kinds. Absent entries pass (they may be created later under the
    /// lock); a symlink or wrong-typed entry fails the open.
    private static func verifyOpenedEntry(path: String,
                                          allowedKinds: Set<Posix.EntryKind>) throws {
        let kind = try Posix.entryKind(path)
        if kind == .absent { return }
        guard allowedKinds.contains(kind) else {
            throw RepackError.installPathUnsafe(
                path: path,
                detail: "unexpected \(kind) entry")
        }
        var info = stat()
        switch kind {
        case .directory:
            let fd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard fd >= 0, fstat(fd, &info) == 0,
                  (info.st_mode & S_IFMT) == S_IFDIR else {
                if fd >= 0 { close(fd) }
                throw RepackError.installPathUnsafe(
                    path: path,
                    detail: "entry was replaced by a non-directory before it could be used")
            }
            close(fd)
        case .regular:
            let fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
            guard fd >= 0, fstat(fd, &info) == 0,
                  (info.st_mode & S_IFMT) == S_IFREG else {
                if fd >= 0 { close(fd) }
                throw RepackError.installPathUnsafe(
                    path: path,
                    detail: "entry was replaced by a non-regular file before it could be used")
            }
            close(fd)
        default:
            throw RepackError.installPathUnsafe(
                path: path,
                detail: "unexpected \(kind) entry")
        }
    }
}
