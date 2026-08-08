import Darwin
import Foundation
import Synchronization
import NVMAI
import NVMAIDecodeProtocol

public final class DecodeServiceInferenceClient: AppModelLifecycleClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, @unchecked Sendable {
    /// Explicit connection lifecycle state: a dead connection is never reused;
    /// the next operation relaunches the service (bounded retry in
    /// `launchIndependentService`).
    private enum ConnectionState {
        case dead
        case connected
    }

    private struct Connection {
        var state: ConnectionState = .dead
        var input: FileHandle?
        var responses: DecodeServiceResponseRouter?
        var launchLabel: String?
        var socketPath: String?
    }

    /// Serializes command writes and carries the load epoch (D6): every unload
    /// bumps the epoch before writing, and a load that was superseded by an
    /// unload aborts without writing its `.load` command.
    private let writeLock = NSLock()
    private var loadEpoch: UInt64 = 0
    private let connection = Mutex(Connection())
    private let serviceURL: URL
    private let inferenceMemory = Mutex<UInt64?>(nil)
    private let activeGenerationID = Mutex<UUID?>(nil)
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    private static let loadEventTimeout: TimeInterval = 30
    private static let generationEventTimeout: TimeInterval = 60
    private static let unloadResponseTimeout: TimeInterval = 30

    public var currentInferenceMemoryBytes: UInt64? {
        inferenceMemory.withLock { $0 }
    }

    public init(serviceURL: URL? = nil) throws {
        if let serviceURL {
            self.serviceURL = serviceURL
        } else if let fallback = Self.resolvedServiceURL() {
            self.serviceURL = fallback
        } else {
            throw DecodeServiceInferenceClientError.serviceURLUnavailable(
                "Neither serviceURL provided nor Bundle.main.executableURL available")
        }
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.loading(.validatingDirectory))
        // Capture the load epoch before launching: a slow launch that was
        // superseded by an unload must not write its `.load` command (D6).
        let epoch = currentLoadEpoch()
        let handles = try await Task.detached(priority: .userInitiated) { [self] in
            try ensureProcess()
        }.value
        let request = DecodeLoadRequest(
            modelPath: modelDirectory.path, maxContextTokens: maxContextTokens,
            runtimeOptions: Self.decodeRuntimeOptions(options),
            forceLogitsHead: forceLogitsHead)
        try writeCommand(DecodeServiceCommand.load(request), epoch: epoch)
        do {
            try await awaitLoadCompletion(handles: handles, request: request, onState: onState)
        } catch {
            if Self.isConnectionError(error) {
                // The service is wedged or dead: mark the connection
                // suspicious so the next attempt relaunches a fresh service
                // instead of reusing a broken one (D2).
                invalidateConnection()
            }
            throw error
        }
    }

    /// Cancels an in-flight service-side load so the service aborts it instead
    /// of only cancelling the app-side wait (D5).
    public func cancelLoad() {
        try? writeCommand(DecodeServiceCommand.cancel(nil))
    }

    public func unload() async {
        let (requestID, handles) = beginUnload()
        // Clear local state regardless of the outcome.
        inferenceMemory.withLock { $0 = nil }
        guard let handles else { return }
        do {
            let event = try await handles.responses.next(
                matching: requestID, timeout: Self.unloadResponseTimeout)
            guard event.kind == .unloaded else {
                invalidateConnection()
                return
            }
        } catch {
            // The decode service did not acknowledge the unload (crashed or
            // wedged). Mark the connection dead so the next operation
            // relaunches a fresh service instead of talking to a broken one.
            invalidateConnection()
        }
    }

    /// Synchronous unload critical section (D6): bumps the load epoch and
    /// writes the unload command atomically under the write lock. Kept out of
    /// the async function so the NSLock is only ever touched outside async
    /// contexts; the caller awaits the response.
    private func beginUnload()
        -> (requestID: UUID, handles: (input: FileHandle, responses: DecodeServiceResponseRouter)?) {
        writeLock.lock()
        defer { writeLock.unlock() }
        loadEpoch &+= 1
        let requestID = UUID()
        let handles = currentHandles()
        if let handles {
            try? handles.input.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceCommand.unload(requestID)))
        }
        return (requestID, handles)
    }

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [self] in
                do {
                    try request.validate()
                    let generationID = UUID()
                    generationTranscriptMailbox.reset()
                    let command = DecodeGenerationRequest(
                        prompt: request.prompt, maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: Self.decodeRuntimeOptions(request.runtimeOptions),
                        generationID: generationID)

                    var didRecover = false
                    while true {
                        do {
                            try await runGenerationSession(
                                request: request,
                                command: command,
                                generationID: generationID,
                                continuation: continuation)
                            break
                        } catch let error {
                            guard !didRecover, Self.isConnectionError(error) else {
                                throw error
                            }
                            // The service crashed or stopped responding
                            // (router EOF or response timeout). Invalidate the
                            // dead connection, re-launch the service once
                            // (bounded retry), re-load the model, and resume
                            // the generation (D1/D2).
                            didRecover = true
                            invalidateConnection()
                            generationTranscriptMailbox.reset()
                            try await reconnectForResume(request: request)
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [weak self] _ in
                task.cancel()
                self?.cancel()
            }
        }
    }

    public func cancel() {
        // Target the active generation so a late cancel never hits a
        // generation that started after the original one ended (D7). Writes
        // are serialized with the other commands so cancel frames can never
        // interleave with an in-flight load/generate frame.
        let generationID = activeGenerationID.withLock { $0 }
        try? writeCommand(DecodeServiceCommand.cancel(generationID))
    }

    deinit {
        let state = connection.withLock { value -> Connection in
            defer { value = Connection() }
            return value
        }
        if state.state == .connected {
            if let input = state.input {
                try? input.write(contentsOf: DecodeFrameCodec.encode(
                    DecodeServiceCommand.shutdown))
                try? input.close()
            }
            Self.tearDownService(label: state.launchLabel, socketPath: state.socketPath)
        }
    }

    // MARK: - Connection lifecycle

    private func ensureProcess() throws
        -> (input: FileHandle, responses: DecodeServiceResponseRouter) {
        if let handles = currentHandles() { return handles }
        return try launchIndependentService()
    }

    private func launchIndependentService() throws
        -> (input: FileHandle, responses: DecodeServiceResponseRouter) {
        guard FileManager.default.isExecutableFile(atPath: serviceURL.path) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service executable is missing at \(serviceURL.path); run swift build -c release before launching the app")
        }
        let token = String(format: "%08x", UInt32.random(in: .min ... .max))
        let label = "com.nvmai.decode.\(getuid()).\(getpid()).\(token)"
        let socketDirectory = try Self.socketDirectory()
        let socketPath = socketDirectory
            .appendingPathComponent("\(getpid()).\(token).sock").path
        guard socketPath.utf8.count < DecodeUnixSocket.sunPathCapacity else {
            throw AppInferenceError.modelLoadFailed(
                "decode service socket path exceeds AF_UNIX limit (\(socketPath.utf8.count) >= \(DecodeUnixSocket.sunPathCapacity))")
        }
        let propertyListURL = URL(
            fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("\(label).plist")
        let propertyList: [String: Any] = [
            "Label": label,
            "ProgramArguments": [
                serviceURL.path,
                "--socket", socketPath,
                "--launch-label", label,
            ],
            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Interactive",
        ]
        let propertyListData = try PropertyListSerialization.data(
            fromPropertyList: propertyList, format: .xml, options: 0)
        try propertyListData.write(to: propertyListURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: propertyListURL) }

        let launcher = Process()
        let errors = Pipe()
        launcher.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        launcher.arguments = [
            "bootstrap", "gui/\(getuid())", propertyListURL.path,
        ]
        launcher.standardOutput = FileHandle.nullDevice
        launcher.standardError = errors
        try launcher.run()
        launcher.waitUntilExit()
        guard launcher.terminationStatus == 0 else {
            let data = try? errors.fileHandleForReading.readToEnd()
            let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let message = detail.flatMap { $0.isEmpty ? nil : $0 }
                ?? "launchd could not start the decode service"
            throw AppInferenceError.modelLoadFailed(message)
        }

        var lastError: Error?
        for _ in 0..<200 {
            do {
                let handles = try DecodeUnixSocket.connect(path: socketPath)
                let responses = DecodeServiceResponseRouter(output: handles.output)
                connection.withLock {
                    $0.state = .connected
                    $0.input = handles.input
                    $0.responses = responses
                    $0.launchLabel = label
                    $0.socketPath = socketPath
                }
                return (handles.input, responses)
            } catch {
                lastError = error
                usleep(10_000)
            }
        }
        // Socket never became ready: boot out the launch job, terminate the
        // service process if it is still alive, and unlink the socket file so
        // no stale entry is left behind (D12).
        Self.tearDownService(label: label, socketPath: socketPath)
        throw AppInferenceError.modelLoadFailed(
            "decode service socket did not become ready: \(lastError.map(String.init(describing:)) ?? "unknown error")")
    }

    /// Re-establishes the service connection after a crash and re-issues the
    /// load so a subsequent generation can resume (D1).
    private func reconnectForResume(request: AppGenerationRequest) async throws {
        let handles = try launchIndependentService()
        let loadRequest = DecodeLoadRequest(
            modelPath: request.modelDirectory.path,
            maxContextTokens: request.maxContextTokens,
            runtimeOptions: Self.decodeRuntimeOptions(request.runtimeOptions),
            forceLogitsHead: !request.isPureGreedy)
        try writeCommand(DecodeServiceCommand.load(loadRequest))
        do {
            try await awaitLoadCompletion(handles: handles, request: loadRequest, onState: nil)
        } catch {
            if Self.isConnectionError(error) {
                invalidateConnection()
            }
            throw error
        }
    }

    /// Consumes `.loading` progress events and returns on `.ready` (or throws
    /// on `.failed`). Applies a per-event timeout so a wedged service fails
    /// the pending call instead of blocking forever (D2).
    private func awaitLoadCompletion(
        handles: (input: FileHandle, responses: DecodeServiceResponseRouter),
        request: DecodeLoadRequest,
        onState: (@Sendable (AppModelLoadState) -> Void)?
    ) async throws {
        while true {
            let event = try await handles.responses.next(
                matching: request.requestID, timeout: Self.loadEventTimeout)
            switch event.kind {
            case .loading:
                if let rawPhase = event.loadPhase,
                   let phase = AppModelLoadPhase(rawValue: rawPhase) {
                    onState?(.loading(phase))
                }
            case .ready:
                inferenceMemory.withLock { $0 = event.currentMemoryBytes }
                onState?(.ready(
                    modelDirectory: URL(fileURLWithPath: request.modelPath),
                    loadSeconds: event.loadSeconds ?? 0))
                return
            case .failed:
                throw AppInferenceError.modelLoadFailed(
                    event.error ?? "decode service load failed")
            default:
                throw AppInferenceError.modelLoadFailed(
                    "decode service returned \(event.kind.rawValue) for a load request")
            }
        }
    }

    /// Runs one generation session over the current connection. Events are
    /// delivered with a 60 s inter-event timeout so a hung service surfaces a
    /// clear error instead of hanging forever (D2). Connection errors are
    /// rethrown so `generate` can reconnect and resume (D1).
    private func runGenerationSession(
        request: AppGenerationRequest,
        command: DecodeGenerationRequest,
        generationID: UUID,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation
    ) async throws {
        guard let handles = currentHandles() else {
            throw AppInferenceError.modelNotLoaded
        }
        activeGenerationID.withLock { $0 = generationID }
        defer {
            activeGenerationID.withLock { current in
                if current == generationID { current = nil }
            }
        }
        try writeCommand(DecodeServiceCommand.generate(command))

        var expectedSequence: UInt64 = 1
        var lastMetricYield = Date.distantPast
        var hasYieldedVisibleText = false
        var pendingText = ""
        while true {
            let event = try await handles.responses.next(
                matching: generationID, timeout: Self.generationEventTimeout)
            inferenceMemory.withLock { $0 = event.currentMemoryBytes }
            guard event.generationID == generationID else { continue }

            if event.kind == .prefill || event.kind == .snapshot {
                guard event.sequence == expectedSequence else {
                    throw AppInferenceError.unknown(
                        "decode service event sequence changed from \(expectedSequence) to \(event.sequence)")
                }
                expectedSequence &+= 1
            }
            if event.kind == .prefill,
               let done = event.prefillDone,
               let total = event.prefillTotal {
                continuation.yield(.prefillProgress(done: done, total: total))
                continue
            }
            if event.kind == .snapshot {
                generationTranscriptMailbox.append(event.textDelta)
                let now = Date()
                let beginsVisibleText = !hasYieldedVisibleText
                    && event.textDelta.contains { !$0.isWhitespace }
                if beginsVisibleText
                    || now.timeIntervalSince(lastMetricYield) >= 0.5 {
                    // Content is never dropped (D16): whitespace-only deltas
                    // are accumulated and flushed together with the next
                    // visible delta so `outputText` keeps the full transcript.
                    // The 0.5 s window only throttles metric/rate updates.
                    lastMetricYield = now
                    hasYieldedVisibleText = hasYieldedVisibleText || beginsVisibleText
                    let accumulated = pendingText + event.textDelta
                    pendingText = ""
                    continuation.yield(.token(AppTokenEvent(
                        index: max(0, event.tokenCount - 1),
                        textDelta: accumulated,
                        elapsedDecodeSeconds: event.decodeSeconds)))
                } else {
                    pendingText += event.textDelta
                }
                continue
            }

            let diagnostics = Self.diagnostics(event, options: request.runtimeOptions)
            switch event.kind {
            case .finished:
                continuation.yield(.finished(diagnostics))
                continuation.finish()
            case .cancelled:
                continuation.yield(.cancelled(diagnostics))
                continuation.finish()
            case .failed:
                let error = AppInferenceError.unknown(
                    event.error ?? "decode service failed")
                continuation.yield(.failed(error, partial: diagnostics))
                continuation.finish(throwing: error)
            default:
                continue
            }
            return
        }
    }

    /// Marks the connection dead and tears the service down so the next
    /// operation relaunches a fresh instance.
    private func invalidateConnection() {
        let state = connection.withLock { value -> Connection in
            defer { value = Connection() }
            return value
        }
        if state.state == .connected {
            Self.tearDownService(label: state.launchLabel, socketPath: state.socketPath)
        }
    }

    private func currentHandles()
        -> (input: FileHandle, responses: DecodeServiceResponseRouter)? {
        connection.withLock { state in
            guard state.state == .connected,
                  let input = state.input,
                  let responses = state.responses else {
                return nil
            }
            return (input, responses)
        }
    }

    private func currentLoadEpoch() -> UInt64 {
        writeLock.lock()
        defer { writeLock.unlock() }
        return loadEpoch
    }

    /// Serialized command writer (D6). When an epoch is supplied, the command
    /// is dropped if an unload has superseded it, so a cancelled load task can
    /// never write its `.load` after an `.unload`.
    private func writeCommand(_ command: DecodeServiceCommand,
                              epoch: UInt64? = nil) throws {
        writeLock.lock()
        defer { writeLock.unlock() }
        if let epoch, epoch != loadEpoch {
            throw CancellationError()
        }
        guard let input = currentHandles()?.input else {
            throw AppInferenceError.modelNotLoaded
        }
        try input.write(contentsOf: DecodeFrameCodec.encode(command))
    }

    private static func isConnectionError(_ error: Error) -> Bool {
        switch error {
        case DecodeFrameError.unexpectedEOF,
             DecodeFrameError.invalidHeader,
             DecodeFrameError.timedOut:
            return true
        default:
            return false
        }
    }

    // MARK: - Socket hygiene (D3, D12)

    private static func socketDirectory() throws -> URL {
        let directory = URL(fileURLWithPath: "/tmp/nvmai-\(getuid())", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: directory.path)
        return directory
    }

    private static func tearDownService(label: String?, socketPath: String?) {
        guard let label else { return }
        // Capture the PID before bootout so a service that survives the
        // bootout (slow startup, wedged accept loop) can still be killed.
        let pid = pidOfLaunchJob(label: label)
        removeLaunchJob(label: label)
        if let pid, kill(pid, 0) == 0 {
            _ = kill(pid, SIGKILL)
        }
        if let socketPath { unlink(socketPath) }
    }

    private static func pidOfLaunchJob(label: String) -> pid_t? {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["print", "gui/\(getuid())/\(label)"]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
              let data = try? output.fileHandleForReading.readToEnd(),
              let text = String(data: data, encoding: .utf8) else {
            return nil
        }
        guard let range = text.range(of: "pid = ") else { return nil }
        let digits = text[range.upperBound...].prefix { $0.isNumber }
        guard !digits.isEmpty else { return nil }
        return pid_t(String(digits))
    }

    // MARK: - Diagnostics mapping

    private static func diagnostics(_ event: DecodeServiceEvent,
                                    options: AppRuntimeOptions) -> AppDiagnostics {
        let stop = AppStopReason(rawValue: event.stopReason ?? "")
            ?? (event.kind == .cancelled
                ? .cancelled
                : event.kind == .failed ? .failed : .maxTokens)
        return AppDiagnostics(
            generatedTokens: event.tokenCount,
            stopReason: stop,
            promptTokenCount: event.promptTokenCount,
            prefillSeconds: event.prefillSeconds,
            timeToFirstTokenSeconds: event.timeToFirstTokenSeconds,
            decodeSeconds: event.decodeSeconds,
            tokensPerSecond: event.tokensPerSecond,
            peakMemoryBytes: event.peakMemoryBytes,
            runtimeOptions: options,
            prefill: prefillDiagnostics(event.prefill, options: options),
            runner: event.runner.map(runnerDiagnostics))
    }

    private static func prefillDiagnostics(
        _ value: DecodePrefillDiagnostics?, options: AppRuntimeOptions
    ) -> PrefillExecutionDiagnostics? {
        guard let value,
              let executedMode = PrefillExecutedMode(rawValue: value.executedMode),
              let completeness = PrefillChunkCompleteness(
                rawValue: value.chunkCompleteness) else { return nil }
        let kvStorage = value.kvStorageMode.flatMap(PrefillKVStorageMode.init(rawValue:))
        return PrefillExecutionDiagnostics(
            config: options.prefillConfig,
            executedMode: executedMode,
            kvStorageMode: kvStorage,
            chunkCompleteness: completeness,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: DecodeRunnerDiagnostics)
        -> AppRunnerDiagnostics {
        AppRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures)
    }

    private static func decodeRuntimeOptions(_ options: AppRuntimeOptions)
        -> DecodeRuntimeOptions {
        DecodeRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: options.expertCachePolicy.rawValue,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: options.rdadvisePolicy.rawValue,
            modelVerification: options.modelVerification.rawValue)
    }

    private static func removeLaunchJob(label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
    }

    private static func resolvedServiceURL() -> URL? {
        guard let executableURL = Bundle.main.executableURL else { return nil }
        return executableURL
            .deletingLastPathComponent()
            .appendingPathComponent("NVMAIDecodeService")
    }
}

public enum DecodeServiceInferenceClientError: Error, LocalizedError {
    case serviceURLUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .serviceURLUnavailable(let reason):
            return "Decode service URL unavailable: \(reason)"
        }
    }
}
