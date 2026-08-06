import Foundation
import Testing
@testable import NVMAIAppCore
@testable import NVMAIDecodeService
import NVMAIDecodeProtocol

@Suite struct DecodeServiceOutboxTests {
    @Test func cancellationFollowedByThrownCancellationWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .cancelled(diagnostics(stopReason: .cancelled)),
            finishError: AppInferenceError.cancelled)

        #expect(event.kind == .cancelled)
        #expect(event.generationID == generationID)
    }

    @Test func failureFollowedByThrownErrorWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .failed(.unknown("first"), partial: nil),
            finishError: AppInferenceError.unknown("second"))

        #expect(event.kind == .failed)
        #expect(event.generationID == generationID)
        #expect(event.error == "first")
    }

    private func firstTerminal(
        from outbox: DecodeServiceOutbox,
        published event: AppInferenceEvent,
        finishError: Error
    ) throws -> DecodeServiceEvent {
        let pipe = Pipe()
        let completion = WriterCompletion()
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                completion.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        outbox.publish(event)
        let terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        outbox.finish(error: finishError)

        // Deterministic: block until the writer thread signals the terminal
        // write. The watchdog only guards against a genuinely hung writer
        // (fail loudly), never a fixed race window.
        let writerFinished = completion.waitForCompletion()
        #expect(writerFinished, "writer did not finish after the terminal event")
        #expect(pipe.fileHandleForReading.readDataToEndOfFile().isEmpty)
        return terminal
    }

    private func diagnostics(stopReason: AppStopReason) -> AppDiagnostics {
        AppDiagnostics(
            generatedTokens: 0,
            stopReason: stopReason,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
    }
}

/// Signals writer completion via NSCondition so the test blocks on the
/// terminal event instead of a fixed sleep/semaphore race window.
private final class WriterCompletion: @unchecked Sendable {
    private let condition = NSCondition()
    private var finished = false

    func signal() {
        condition.lock()
        finished = true
        condition.broadcast()
        condition.unlock()
    }

    func waitForCompletion(watchdog: TimeInterval = 10) -> Bool {
        condition.lock()
        while !finished {
            if !condition.wait(until: Date().addingTimeInterval(watchdog)) { break }
        }
        condition.unlock()
        return finished
    }
}
