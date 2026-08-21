import Foundation

public enum ExpertLoadOperationState: Sendable, Equatable {
    case submitted
    case inFlight
    case completed
    case failed
}

/// One routed-expert storage batch whose submission is independent from its
/// completion. The operation is deliberately backend-neutral: pread and Metal
/// I/O publish the same state and error contract to the decode scheduler.
///
/// unchecked-invariant: all mutable state and continuations are guarded by
/// `condition`; terminal transition happens exactly once.
public final class ExpertLoadOperation: @unchecked Sendable {
    private let condition = NSCondition()
    private var currentState: ExpertLoadOperationState = .submitted
    private var failure: (any Error)?
    private var continuations: [CheckedContinuation<Void, any Error>] = []

    public let submittedNanos: UInt64
    public let completionToken: ExpertIOCompletionToken?
    /// Present only for the Metal-I/O staging route. The dependent compute
    /// command owns the copy into the expert cache and must release it after
    /// that command completes.
    let metalStagingTransfer: MetalExpertStagingTransfer?
    /// A staged Metal load is not resident until the event-gated GPU blit has
    /// completed. Pread writes cache slots directly and therefore remains
    /// false.
    let requiresGPUFinalization: Bool
    private var startedAtNanos: UInt64 = 0
    private var completedAtNanos: UInt64 = 0

    private let eventCoordinator: ExpertIOEventCoordinator?
    private let backendSignalsEvent: Bool

    init(completionToken: ExpertIOCompletionToken? = nil,
         eventCoordinator: ExpertIOEventCoordinator? = nil,
         backendSignalsEvent: Bool = false,
         metalStagingTransfer: MetalExpertStagingTransfer? = nil,
         requiresGPUFinalization: Bool = false,
         submittedNanos: UInt64 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) {
        precondition(!requiresGPUFinalization || metalStagingTransfer != nil,
                     "a staged Metal load requires a staging transfer")
        self.completionToken = completionToken
        self.eventCoordinator = eventCoordinator
        self.backendSignalsEvent = backendSignalsEvent
        self.metalStagingTransfer = metalStagingTransfer
        self.requiresGPUFinalization = requiresGPUFinalization
        self.submittedNanos = submittedNanos
    }

    func releaseStagingTransfer() { metalStagingTransfer?.release() }

    public var state: ExpertLoadOperationState {
        condition.withLock { currentState }
    }

    public var submissionToStartNanos: UInt64 {
        condition.withLock {
            startedAtNanos >= submittedNanos ? startedAtNanos - submittedNanos : 0
        }
    }

    public var loadNanos: UInt64 {
        condition.withLock {
            completedAtNanos >= startedAtNanos ? completedAtNanos - startedAtNanos : 0
        }
    }

    public var startedNanos: UInt64 { condition.withLock { startedAtNanos } }
    public var completedNanos: UInt64 { condition.withLock { completedAtNanos } }

    /// Blocking compatibility wait for tests and the v4.1 fallback path.
    public func wait() throws {
        condition.lock()
        while currentState == .submitted || currentState == .inFlight {
            condition.wait()
        }
        let error = failure
        condition.unlock()
        if let error { throw error }
    }

    /// Suspension-only completion for schedulers. No worker thread is occupied
    /// while the storage service owns the request.
    public func completion() async throws {
        try await withCheckedThrowingContinuation { continuation in
            condition.lock()
            switch currentState {
            case .completed:
                condition.unlock()
                continuation.resume()
            case .failed:
                let error = failure!
                condition.unlock()
                continuation.resume(throwing: error)
            case .submitted, .inFlight:
                continuations.append(continuation)
                condition.unlock()
            }
        }
    }

    func markInFlight() {
        condition.withLock {
            precondition(currentState == .submitted,
                         "expert load operation started more than once")
            currentState = .inFlight
            startedAtNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        }
    }

    func finish(_ result: Result<Void, any Error>) {
        condition.lock()
        precondition(currentState == .submitted || currentState == .inFlight,
                     "expert load operation completed more than once")
        completedAtNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        switch result {
        case .success:
            currentState = .completed
        case .failure(let error):
            currentState = .failed
            failure = error
        }
        let waiting = continuations
        continuations.removeAll(keepingCapacity: false)
        condition.broadcast()
        condition.unlock()

        if let completionToken, let eventCoordinator {
            if backendSignalsEvent {
                eventCoordinator.recordBackendSignal(completionToken)
            } else {
                switch result {
                case .success:
                    eventCoordinator.publish(completionToken, succeeded: true)
                case .failure:
                    // Failure still advances the timeline so a pre-submitted GPU
                    // command cannot deadlock. Event-aware kernels see status=2
                    // and avoid dereferencing the incomplete expert slots.
                    eventCoordinator.publish(completionToken, succeeded: false)
                }
            }
        }

        for continuation in waiting {
            switch result {
            case .success:
                continuation.resume()
            case .failure(let error):
                continuation.resume(throwing: error)
            }
        }
    }
}

/// Persistent bounded submission service. The C pread backend owns the actual
/// fixed reader threads; these queues replace per-layer use of the process-wide
/// global queue and bound the number of batches entering those pools.
/// unchecked-invariant: worker selection is locked and queues are immutable.
final class ExpertIOScheduler: @unchecked Sendable {
    static let shared = ExpertIOScheduler(workerCount: 4)

    private let workers: [DispatchQueue]
    private let selectionLock = NSLock()
    private var nextWorker = 0

    init(workerCount: Int) {
        precondition(workerCount > 0)
        workers = (0..<workerCount).map { index in
            DispatchQueue(label: "NVMAI.expert-io.\(index)", qos: .userInitiated)
        }
    }

    func submit(_ work: @escaping @Sendable () -> Void) {
        selectionLock.lock()
        let worker = workers[nextWorker]
        nextWorker = (nextWorker + 1) % workers.count
        selectionLock.unlock()
        worker.async(execute: work)
    }
}
