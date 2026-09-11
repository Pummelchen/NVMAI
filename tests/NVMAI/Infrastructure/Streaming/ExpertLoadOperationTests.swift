import Foundation
import Testing

@testable import NVMAI

@Suite struct ExpertLoadOperationTests {
    @Test func operationPublishesStateAndResumesAsyncWaiter() async throws {
        let operation = ExpertLoadOperation()
        #expect(operation.state == .submitted)

        let waiter = Task { try await operation.completion() }
        operation.markInFlight()
        #expect(operation.state == .inFlight)
        operation.finish(.success(()))

        try await waiter.value
        #expect(operation.state == .completed)
        #expect(operation.completedNanos >= operation.startedNanos)
    }

    @Test func operationPropagatesFailureToBlockingAndAsyncWaiters() async {
        struct ExpectedFailure: Error {}
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        operation.finish(.failure(ExpectedFailure()))

        await #expect(throws: ExpectedFailure.self) {
            try await operation.completion()
        }
        #expect(throws: ExpectedFailure.self) { try operation.wait() }
        #expect(operation.state == .failed)
    }

    @Test func sharedEventDoesNotAdvancePastOutOfOrderBatch() throws {
        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let firstToken = try coordinator.reserve()
        let secondToken = try coordinator.reserve()
        let first = ExpertLoadOperation(completionToken: firstToken,
                                        eventCoordinator: coordinator)
        let second = ExpertLoadOperation(completionToken: secondToken,
                                         eventCoordinator: coordinator)

        second.finish(.success(()))
        #expect(secondToken.status.contents().advanced(by: secondToken.statusOffset)
            .load(as: UInt32.self) == 1)
        #expect(secondToken.event.signaledValue == 0)

        first.finish(.success(()))
        #expect(firstToken.event.signaledValue == secondToken.value)
        #expect(firstToken.value != secondToken.value)
    }

    @Test func failedBatchSignalsTerminalStatusInsteadOfDeadlocking() throws {
        struct ExpectedFailure: Error {}
        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let token = try coordinator.reserve()
        let operation = ExpertLoadOperation(completionToken: token,
                                            eventCoordinator: coordinator)

        operation.finish(.failure(ExpectedFailure()))

        #expect(token.status.contents().advanced(by: token.statusOffset)
            .load(as: UInt32.self) == 2)
        #expect(token.event.signaledValue == token.value)
        #expect(throws: ExpectedFailure.self) { try operation.wait() }
    }
}
