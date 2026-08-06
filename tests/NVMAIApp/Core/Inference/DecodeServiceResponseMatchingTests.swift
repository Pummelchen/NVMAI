import Foundation
import Testing
@testable import NVMAIAppCore
import NVMAIDecodeProtocol

@Suite struct DecodeServiceResponseMatchingTests {
    /// All events in these tests are pre-written into an in-memory pipe, so
    /// matching resolves immediately; the timeout is only a safety net against
    /// a dead reader.
    private let responseTimeout: TimeInterval = 10

    @Test func staleGenerationTerminalIsSkippedBeforeLoadResponse() async throws {
        let expectedID = UUID()
        let pipe = try pipe(containing: [
            DecodeServiceEvent(kind: .cancelled, generationID: UUID()),
            DecodeServiceEvent(kind: .ready, generationID: expectedID),
        ])

        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        let event = try await router.next(matching: expectedID, timeout: responseTimeout)

        #expect(event.kind == .ready)
        #expect(event.generationID == expectedID)
    }

    @Test func staleGenerationTerminalIsSkippedBeforeUnloadResponse() async throws {
        let expectedID = UUID()
        let pipe = try pipe(containing: [
            DecodeServiceEvent(kind: .cancelled, generationID: UUID()),
            DecodeServiceEvent(kind: .unloaded, generationID: expectedID),
        ])

        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        let event = try await router.next(matching: expectedID, timeout: responseTimeout)

        #expect(event.kind == .unloaded)
        #expect(event.generationID == expectedID)
    }

    @Test func matchingFailedLoadResponseIsReturned() async throws {
        let expectedID = UUID()
        let pipe = try pipe(containing: [
            DecodeServiceEvent(
                kind: .failed, generationID: expectedID, error: "load failed"),
        ])

        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        let event = try await router.next(matching: expectedID, timeout: responseTimeout)

        #expect(event.kind == .failed)
        #expect(event.error == "load failed")
    }

    @Test func concurrentWaitersReceiveOnlyTheirOwnResponses() async throws {
        let loadID = UUID()
        let unloadID = UUID()
        let pipe = Pipe()
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        async let load = router.next(matching: loadID, timeout: responseTimeout)
        async let unload = router.next(matching: unloadID, timeout: responseTimeout)

        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(kind: .unloaded, generationID: unloadID)))
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(kind: .ready, generationID: loadID)))
        try pipe.fileHandleForWriting.close()

        let (loadEvent, unloadEvent) = try await (load, unload)
        #expect(loadEvent.kind == .ready)
        #expect(loadEvent.generationID == loadID)
        #expect(unloadEvent.kind == .unloaded)
        #expect(unloadEvent.generationID == unloadID)
    }

    @Test func endOfFileBeforeMatchingResponseFails() async throws {
        let pipe = try pipe(containing: [
            DecodeServiceEvent(kind: .cancelled, generationID: UUID()),
        ])
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)

        await #expect(throws: DecodeFrameError.self) {
            _ = try await router.next(matching: UUID(), timeout: responseTimeout)
        }
    }

    private func pipe(containing events: [DecodeServiceEvent]) throws -> Pipe {
        let pipe = Pipe()
        for event in events {
            try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(event))
        }
        try pipe.fileHandleForWriting.close()
        return pipe
    }
}
