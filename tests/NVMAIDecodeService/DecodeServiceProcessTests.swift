import Darwin
import Foundation
import Testing
@testable import NVMAIAppCore
import NVMAIDecodeProtocol

/// Spawns the real `NVMAIDecodeService` binary and drives it over its Unix
/// socket (T31). No real model is required: the test pins the process-level
/// lifecycle that does not touch weights — the socket handshake, a failed
/// load (bad model path) reporting a terminal `.failed` event, the
/// not-loaded `.generate` guard, an idle `.unload`, and clean `.shutdown`
/// exit.
@Suite struct DecodeServiceProcessTests {

    @Test func processHandshakeFailedLoadAndShutdown() throws {
        let binary = try Self.locateServiceBinary()
        let socketDirectory = URL(fileURLWithPath: "/tmp/nvmai-\(getuid())", isDirectory: true)
        try FileManager.default.createDirectory(
            at: socketDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let token = String(format: "%08x", UInt32.random(in: .min ... .max))
        let socketPath = socketDirectory
            .appendingPathComponent("t-\(getpid()).\(token).sock")
            .path
        #expect(socketPath.utf8.count < DecodeUnixSocket.sunPathCapacity)
        defer { try? FileManager.default.removeItem(atPath: socketPath) }

        let process = Process()
        process.executableURL = binary
        process.arguments = ["--socket", socketPath]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        defer {
            if process.isRunning {
                process.terminate()
            }
        }

        // The service creates the socket file when it binds; wait for it so
        // connect() cannot race ENOENT.
        let bindDeadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: socketPath),
              Date() < bindDeadline {
            usleep(10_000)
        }
        #expect(FileManager.default.fileExists(atPath: socketPath),
                "decode service did not create its socket within the deadline")

        let handles = try DecodeUnixSocket.connect(path: socketPath)
        defer {
            try? handles.input.close()
            try? handles.output.close()
        }
        try handles.output.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.load(DecodeLoadRequest(
                modelPath: "/nonexistent/model.gturbo",
                maxContextTokens: 1024))))

        // A load of a missing model fails fast with a terminal `.failed`
        // event (D11 clears session state; the error string names the dir).
        let failedLoad = try Self.readEvent(
            from: handles.input,
            until: { $0.kind == .failed },
            timeout: 10)
        #expect(failedLoad.error?.isEmpty == false)
        #expect(failedLoad.error?.contains("model.gturbo") == true
                || failedLoad.error?.contains("gturbo") == true)

        // The failed load left the session unloaded: a generate is rejected
        // with the not-loaded guard instead of a stale-state crash.
        try handles.output.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.generate(DecodeGenerationRequest(
                prompt: "hi",
                maxNewTokens: 4,
                maxContextTokens: 1024,
                temperature: 0))))
        let notLoaded = try Self.readEvent(
            from: handles.input,
            until: { $0.kind == .failed },
            timeout: 10)
        #expect(notLoaded.error?.contains("model is not loaded") == true)

        // An idle unload is a safe no-op that still reports `.unloaded`.
        let unloadID = UUID()
        try handles.output.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.unload(unloadID)))
        let unloaded = try Self.readEvent(
            from: handles.input,
            until: { $0.kind == .unloaded },
            timeout: 10)
        #expect(unloaded.generationID == unloadID)

        // Shutdown: the service exits and unlinks its socket.
        try handles.output.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.shutdown))
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        #expect(!FileManager.default.fileExists(atPath: socketPath),
                "service must unlink its socket on exit")
    }

    // MARK: - Helpers

    /// The test step builds every executable product in debug, so the binary
    /// lives under `.build/<triple-or-flat>/debug/NVMAIDecodeService`.
    private static func locateServiceBinary() throws -> URL {
        var directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<6 {
            for relative in ["arm64-apple-macosx/debug/NVMAIDecodeService",
                             "debug/NVMAIDecodeService"] {
                let candidate = directory
                    .appendingPathComponent(".build")
                    .appendingPathComponent(relative)
                if FileManager.default.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
            directory.deleteLastPathComponent()
        }
        throw DecodeServiceProcessError.binaryNotFound
    }

    /// Read frames until `predicate` matches (or the deadline passes).
    /// Uses poll() so a stalled service fails the test instead of hanging it.
    private static func readEvent(
        from handle: FileHandle,
        until predicate: (DecodeServiceEvent) -> Bool,
        timeout: TimeInterval
    ) throws -> DecodeServiceEvent {
        let fd = handle.fileDescriptor
        var descriptor = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if Darwin.poll(&descriptor, 1, 100) > 0 {
                let event = try DecodeFrameCodec.read(DecodeServiceEvent.self, from: handle)
                if predicate(event) {
                    return event
                }
            }
        }
        throw DecodeServiceProcessError.timedOut
    }
}

private enum DecodeServiceProcessError: Error, CustomStringConvertible {
    case binaryNotFound
    case timedOut

    var description: String {
        switch self {
        case .binaryNotFound:
            "NVMAIDecodeService debug binary not found; run `swift build --build-tests` first"
        case .timedOut:
            "timed out waiting for the decode service event"
        }
    }
}
