import Darwin
import Foundation
import NVMAI
import NVMAIAppCore
import NVMAIDecodeProtocol

@main enum NVMAIDecodeServiceMain {
    /// lint:allow-long the decode service's command loop: one `case` per
    /// protocol message, each with its reply. Splitting it per command would
    /// hide the exhaustive switch that makes an unhandled message a
    /// compile-visible gap, and the cases share the session state.
    static func main() async {
        let socketPath = argument(after: "--socket")
        let launchLabel = argument(after: "--launch-label")
        let handles: (input: FileHandle, output: FileHandle)
        do {
            handles = if let socketPath {
                try DecodeUnixSocket.listenAndAccept(path: socketPath)
            } else {
                (.standardInput, .standardOutput)
            }
        } catch {
            FileHandle.standardError.write(Data("Decode service transport failed: \(error)\n".utf8))
            Foundation.exit(1)
        }
        defer {
            if let socketPath { unlink(socketPath) }
            if let launchLabel { retireLaunchJob(launchLabel) }
        }

        let client = RealInferenceClient()
        let commands = DecodeCommandQueue()
        let session = ServiceSession()
        let memorySampler = AppMemorySampler()
        let input = Thread {
            do {
                while true {
                    let command = try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handles.input)
                    // D7: cancellation is targeted by generation ID and handled
                    // here so it lands promptly while the main loop is busy in a
                    // generation or load.
                    if case .cancel(let id) = command {
                        handleCancel(id, client: client, session: session)
                    }
                    commands.append(command)
                    if case .shutdown = command { break }
                }
            } catch {
                commands.close()
            }
        }
        input.name = "NVMAI.DecodeService.Input"
        input.qualityOfService = .userInitiated
        input.start()

        var modelDirectory: URL?
        var loadedOptions: DecodeRuntimeOptions?
        while let command = await nextCommand(commands) {
            switch command {
            case .load(let request):
                let directory = URL(fileURLWithPath: request.modelPath)
                let started = Date()
                // D5: the load runs in a cancellable task so an incoming
                // `.cancel` can abort it on the service side. Progress phases
                // are streamed to the app (D10); the task also emits a
                // heartbeat so the app-side load-phase timeout never fires
                // during a long verification.
                let loadTask = Task { () -> LoadOutcome in
                    let progress = LoadPhaseReporter()
                    let heartbeat = Task {
                        while !Task.isCancelled {
                            try? await Task.sleep(for: .seconds(10))
                            guard !Task.isCancelled, let phase = progress.phaseLabel else {
                                continue
                            }
                            writeBestEffort(DecodeServiceEvent(
                                kind: .loading, generationID: request.requestID,
                                loadPhase: phase), to: handles.output)
                        }
                    }
                    defer { heartbeat.cancel() }
                    do {
                        let options = try appRuntimeOptions(request.runtimeOptions)
                        try Task.checkCancellation()
                        try await client.ensureLoaded(
                            modelDirectory: directory,
                            maxContextTokens: request.maxContextTokens,
                            options: options,
                            forceLogitsHead: request.forceLogitsHead) { state in
                            switch state {
                            case .loading(let phase):
                                progress.phaseLabel = phase.label
                                writeBestEffort(DecodeServiceEvent(
                                    kind: .loading, generationID: request.requestID,
                                    loadPhase: phase.label), to: handles.output)
                            case .ready, .failed, .notLoaded, .cancelling, .unloading:
                                break
                            }
                        }
                        try Task.checkCancellation()
                        return .success(Date().timeIntervalSince(started))
                    } catch is CancellationError {
                        return .cancelled
                    } catch {
                        return .failure(error)
                    }
                }
                session.loadTask = loadTask
                switch await loadTask.value {
                case .success(let loadSeconds):
                    modelDirectory = directory
                    loadedOptions = request.runtimeOptions
                    let memory = memorySampler.sample()
                    writeBestEffort(DecodeServiceEvent(
                        kind: .ready, generationID: request.requestID,
                        loadSeconds: loadSeconds,
                        currentMemoryBytes: memory, peakMemoryBytes: memory),
                        to: handles.output)
                case .cancelled:
                    // D11: a cancelled load must not leave stale session state.
                    modelDirectory = nil
                    loadedOptions = nil
                    writeBestEffort(DecodeServiceEvent(
                        kind: .failed, generationID: request.requestID,
                        error: "model load cancelled"), to: handles.output)
                case .failure(let error):
                    // D11: clear stale state and emit a `failed` event so the
                    // app resets its load state.
                    modelDirectory = nil
                    loadedOptions = nil
                    writeBestEffort(DecodeServiceEvent(
                        kind: .failed, generationID: request.requestID,
                        error: "\(error)"), to: handles.output)
                }
                session.loadTask = nil
            case .generate(let request):
                guard let modelDirectory else {
                    writeBestEffort(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "model is not loaded"), to: handles.output)
                    continue
                }
                guard request.runtimeOptions == loadedOptions else {
                    writeBestEffort(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "generation runtime options do not match the loaded session"),
                        to: handles.output)
                    continue
                }

                let outbox = DecodeServiceOutbox(generationID: request.generationID)
                let writerFinished = DispatchSemaphore(value: 0)
                let writer = Thread {
                    defer { writerFinished.signal() }
                    do { try outbox.runWriter(to: handles.output) }
                    catch {
                        FileHandle.standardError.write(Data("IPC writer failed: \(error)\n".utf8))
                    }
                }
                writer.name = "NVMAI.DecodeService.Writer"
                writer.qualityOfService = .userInitiated
                writer.start()

                session.activeGenerationID = request.generationID
                do {
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    let generation = AppGenerationRequest(
                        modelDirectory: modelDirectory, prompt: request.prompt,
                        maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        topK: request.topK,
                        topP: request.topP,
                        presencePenalty: request.presencePenalty,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: options)
                    for try await event in client.generate(generation) {
                        outbox.publish(event)
                    }
                    outbox.finish()
                } catch {
                    outbox.finish(error: error)
                }
                session.activeGenerationID = nil
                await withCheckedContinuation { continuation in
                    DispatchQueue.global(qos: .userInitiated).async {
                        writerFinished.wait()
                        continuation.resume()
                    }
                }
            case .cancel(let id):
                // Re-affirm the cancel once the current operation completes: a
                // `.cancel` that arrived before the load task was registered is
                // applied here (the input thread already handled the prompt
                // case).
                handleCancel(id, client: client, session: session)
            case .unload(let requestID):
                await client.unload()
                modelDirectory = nil
                loadedOptions = nil
                writeBestEffort(DecodeServiceEvent(
                    kind: .unloaded, generationID: requestID), to: handles.output)
            case .shutdown:
                await client.unload()
                return
            }
        }
    }

    // MARK: - Session state shared between the input thread and the main loop

    private enum LoadOutcome {
        case success(Double)
        case cancelled
        case failure(Error)
    }

    /// unchecked-invariant: the single backing field is guarded by `lock`. The
    /// load task writes the phase while the command loop reads it to answer
    /// status queries.
    private final class LoadPhaseReporter: @unchecked Sendable {
        private let lock = NSLock()
        private var _phaseLabel: String?

        var phaseLabel: String? {
            get { lock.lock(); defer { lock.unlock() }; return _phaseLabel }
            set { lock.lock(); _phaseLabel = newValue; lock.unlock() }
        }
    }

    /// unchecked-invariant: every stored property is guarded by `lock`. The
    /// session is shared between the command loop and the in-flight load and
    /// generation tasks, which is why the accessors are lock-wrapped rather
    /// than the type being an actor -- the command loop needs synchronous reads.
    private final class ServiceSession: @unchecked Sendable {
        private let lock = NSLock()
        private var _activeGenerationID: UUID?
        private var _loadTask: Task<LoadOutcome, Never>?

        var activeGenerationID: UUID? {
            get { lock.lock(); defer { lock.unlock() }; return _activeGenerationID }
            set { lock.lock(); _activeGenerationID = newValue; lock.unlock() }
        }

        var loadTask: Task<LoadOutcome, Never>? {
            get { lock.lock(); defer { lock.unlock() }; return _loadTask }
            set { lock.lock(); _loadTask = newValue; lock.unlock() }
        }
    }

    /// D7: cancel the generation whose ID matches, or — for an untargeted
    /// cancel — the active generation, or the in-flight load when no
    /// generation is running.
    private static func handleCancel(_ id: UUID?,
                                     client: RealInferenceClient,
                                     session: ServiceSession) {
        if let id {
            if session.activeGenerationID == id {
                client.cancel()
            }
        } else {
            if session.activeGenerationID != nil {
                client.cancel()
            }
            session.loadTask?.cancel()
        }
    }

    private static func nextCommand(_ commands: DecodeCommandQueue)
        async -> DecodeServiceCommand? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: commands.next())
            }
        }
    }

    /// Serializes every frame written by the main loop, the load heartbeat,
    /// and the load-phase progress callback so two concurrent writers can
    /// never interleave bytes of a single frame on the socket.
    private static let outputLock = NSLock()

    private static func writeBestEffort(_ event: DecodeServiceEvent,
                                        to handle: FileHandle) {
        outputLock.lock()
        defer { outputLock.unlock() }
        try? handle.write(contentsOf: DecodeFrameCodec.encode(event))
    }

    private static func appRuntimeOptions(_ options: DecodeRuntimeOptions) throws
        -> AppRuntimeOptions {
        guard let cachePolicy = AppExpertCachePolicy(
            rawValue: options.expertCachePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown expert cache policy \(options.expertCachePolicy)")
        }
        guard let rdadvisePolicy = AppRDAdvicePolicy(
            rawValue: options.rdadvisePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown RDADVISE policy \(options.rdadvisePolicy)")
        }
        guard let modelVerification = AppModelVerification(
            rawValue: options.modelVerification) else {
            throw AppInferenceError.invalidRequest(
                "unknown model verification \(options.modelVerification)")
        }
        guard let kvCachePrecision = KVCachePrecision(rawValue: options.kvCacheBits) else {
            throw AppInferenceError.invalidRequest(
                "unknown KV-cache precision \(options.kvCacheBits)")
        }
        guard let ropeScalingMode = RuntimeRoPEScalingMode(
            rawValue: options.ropeScalingMode) else {
            throw AppInferenceError.invalidRequest(
                "unknown RoPE scaling mode \(options.ropeScalingMode)")
        }
        let resolved = AppRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: cachePolicy,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: rdadvisePolicy,
            modelVerification: modelVerification,
            conciseMode: options.conciseMode,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode)
        try resolved.validate()
        return resolved
    }

    private static func argument(after name: String) -> String? {
        let arguments = CommandLine.arguments
        guard let index = arguments.firstIndex(of: name),
              arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }

    private static func retireLaunchJob(_ label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
    }
}
