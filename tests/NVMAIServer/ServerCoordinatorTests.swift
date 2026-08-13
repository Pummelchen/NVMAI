import Testing
@testable import NVMAIServerCore

private actor TestGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

@Suite("Server coordinator")
struct ServerCoordinatorTests {
    @Test func boundsFIFOAndRecoversAfterCancellation() async throws {
        let coordinator = ServerCoordinator(queueLimit: 1)
        let gate = TestGate()
        let active = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.isActive }
        let queued = Task {
            try await coordinator.run { 2 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 1 }
        await #expect(throws: ServerRequestError.queueFull) {
            try await coordinator.run { 3 }
        }
        queued.cancel()
        _ = try? await queued.value
        await gate.open()
        #expect(try await active.value == 1)
        #expect(await coordinator.queuedCount == 0)
    }

    /// Bounded poll so a state that never reaches `condition` fails fast
    /// instead of spinning forever.
    private func waitUntil(timeout: Duration,
                           _ condition: () async -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while await !condition() {
            guard ContinuousClock.now < deadline else {
                throw CoordinatorTimeout()
            }
            await Task.yield()
        }
    }

    private struct CoordinatorTimeout: Error {}
}
