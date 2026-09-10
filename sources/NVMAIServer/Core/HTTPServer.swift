import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import NVMAI

public actor NVMAIHTTPServer {
    public static let maximumBodyBytes = 1_048_576

    /// S1: close a connection that has been idle (no request in flight) for
    /// this long. Also used as the pipeline read timeout, bounding slowloris
    /// style partial-request stalls.
    public static let idleReadTimeout: TimeAmount = .seconds(120)

    /// S1: reject connections beyond this cap to bound FD/memory usage.
    public static let maximumConcurrentConnections = 64

    private let group: MultiThreadedEventLoopGroup
    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let reasoningProfile: ServerReasoningProfile
    /// Set when the server routes between catalog models; nil keeps the
    /// single-model server exactly as it was.
    private let router: (any ModelRouting)?
    private let childChannels = ChildChannelRegistry(
        maximumChannels: maximumConcurrentConnections)
    /// Finished /v1/responses kept for previous_response_id and retrieval.
    private let responseStore = ResponseStore()
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    public init(modelID: String,
                queueLimit: Int,
                backend: any ServerInferenceBackend,
                heartbeatInterval: TimeAmount = .seconds(5),
                reasoningProfile: ServerReasoningProfile = .default,
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1),
                router: (any ModelRouting)? = nil) {
        self.group = group
        self.modelID = modelID
        self.backend = backend
        self.coordinator = ServerCoordinator(queueLimit: queueLimit)
        self.heartbeatInterval = heartbeatInterval
        self.reasoningProfile = reasoningProfile
        self.router = router
    }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let backend = self.backend
        let coordinator = self.coordinator
        let heartbeatInterval = self.heartbeatInterval
        let reasoningProfile = self.reasoningProfile
        let router = self.router
        let childChannels = self.childChannels
        let responseStore = self.responseStore
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                return channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true
                ).flatMap {
                    channel.pipeline.addHandler(ServerHTTPHandler(
                        modelID: modelID,
                        backend: backend,
                        coordinator: coordinator,
                        heartbeatInterval: heartbeatInterval,
                        reasoningProfile: reasoningProfile,
                        router: router,
                        childChannels: childChannels,
                        responseStore: responseStore))
                }
            }
            // S29: so_reuseaddr belongs on the listening socket only, not on
            // accepted sockets.
        let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
        self.channel = channel
        return channel
    }

    public func shutdown() async throws {
        if let shutdownTask {
            try await shutdownTask.value
            return
        }

        let listeningChannel = channel
        channel = nil
        let childChannels = self.childChannels
        let coordinator = self.coordinator
        let group = self.group
        let task = Task { @Sendable in
            var firstError: (any Error)?
            await coordinator.shutdown()
            if let listeningChannel {
                do {
                    try await listeningChannel.close().get()
                } catch ChannelError.alreadyClosed {
                } catch {
                    firstError = error
                }
            }
            await childChannels.closeAll()
            do {
                try await group.shutdownGracefully()
            } catch {
                if firstError == nil {
                    firstError = error
                }
            }
            if let firstError {
                throw firstError
            }
        }
        shutdownTask = task
        try await task.value
    }

    var queuedRequestCount: Int {
        get async { await coordinator.queuedCount }
    }

    var hasActiveRequest: Bool {
        get async { await coordinator.isActive }
    }

    var acceptedConnectionCount: Int {
        childChannels.count
    }
}

/// The workspace a request names, from `X-NVMAI-Workspace`.
///
/// This is how one server serves several checkouts, and it was documented in
/// three places and read in none: `ValidatedChatRequest.workspace` had no
/// caller at all, so two projects sharing a server silently shared a memory
/// store — the exact cross-project mixing the workspace design exists to
/// prevent.
///
/// Only the shape is checked here. Whether the name is *allowed* — the
/// reserved shared workspace, characters a scope forbids — belongs to the
/// memory layer, which already refuses those and disables memory for the
/// request rather than failing it. A bad workspace must never cost someone
/// their answer.
enum WorkspaceHeader {
    static let name = "x-nvmai-workspace"

    static func value(in head: HTTPRequestHead?) -> String? {
        guard let raw = head?.headers.first(name: name) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        // A proxy that adds the header with nothing after it must not create
        // a workspace called "".
        return trimmed.isEmpty ? nil : trimmed
    }
}

/// unchecked-invariant: NIO calls every ChannelInboundHandler method on the
/// channel's own event loop, so the handler's per-request state is already
/// serialised. The exceptions are `activeTask`, which the SSE drainer and the
/// backpressure path touch from the cooperative pool -- that field is guarded by
/// `taskLock` -- and `responseModelID`, which the response builders read from
/// the generation task and which is guarded by `responseModelLock`.
private final class ServerHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    /// The OpenAI REST protocol/version identifier reported in the
    /// `openai-version` response header on every API response. The REST API
    /// major version is v1 (endpoints under /v1/...).
    private static let openAIVersionHeader = ("openai-version", "2020-10-01")

    /// S4: cap on SSE frames waiting to be written; a slow reader that exceeds
    /// it fails the stream instead of growing the pending queue without bound.
    private static let maximumPendingStreamChunks = 512

    private static let minimalErrorData = Data(#"""
    {"error":{"message":"internal server error","type":"server_error","code":"internal_error"}}
    """#.utf8)

    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let reasoningProfile: ServerReasoningProfile
    private let router: (any ModelRouting)?
    private let childChannels: ChildChannelRegistry
    private let responseStore: ResponseStore
    private var head: HTTPRequestHead?

    /// The model the current request was validated for, echoed in every
    /// response object it produces. One request is in flight per connection
    /// (pipelining assistance holds the next head until this response ends),
    /// but the echo sites run on the cooperative pool, hence the lock.
    private let responseModelLock = NSLock()
    private var _responseModelID: String
    private var responseModelID: String {
        get { responseModelLock.withLock { _responseModelID } }
        set { responseModelLock.withLock { _responseModelID = newValue } }
    }
    private var body = ByteBuffer()
    private var oversized = false

    // Access to activeTask is lock-guarded because the SSE drainer and the
    // backpressure fail path read it from the cooperative pool while the event
    // loop writes it for each new request.
    private let taskLock = NSLock()
    private var _activeTask: Task<Void, Never>?
    private var activeTask: Task<Void, Never>? {
        get { taskLock.withLock { _activeTask } }
        set { taskLock.withLock { _activeTask = newValue } }
    }

    private var requestPhaseState = RequestPhaseState()
    /// Requests currently being processed on this connection; idle closing and
    /// phase bookkeeping key off it (S10, S25, S1).
    private var inFlightRequests = 0
    private var idleCloseTask: Scheduled<Void>?

    init(modelID: String,
         backend: any ServerInferenceBackend,
         coordinator: ServerCoordinator,
         heartbeatInterval: TimeAmount,
         reasoningProfile: ServerReasoningProfile,
         router: (any ModelRouting)?,
         childChannels: ChildChannelRegistry,
         responseStore: ResponseStore) {
        self.modelID = modelID
        self._responseModelID = modelID
        self.router = router
        self.reasoningProfile = reasoningProfile
        self.backend = backend
        self.coordinator = coordinator
        self.heartbeatInterval = heartbeatInterval
        self.childChannels = childChannels
        self.responseStore = responseStore
    }

    /// Which API's wire shapes a request speaks. Error envelopes, stream
    /// terminators and heartbeat frames differ per surface; the generation
    /// underneath does not.
    private enum APISurface: Sendable {
        case chat, responses, anthropic
    }

    func channelActive(context: ChannelHandlerContext) {
        // S1: start the per-connection idle deadline so a connection that
        // never sends anything is closed.
        resetIdleDeadline(context)
        context.fireChannelActive()
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        // S1: any read activity pushes the idle deadline out (slowloris
        // connections stall once the trickle stops).
        resetIdleDeadline(context)
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            body.clear()
            oversized = false
            // S10/S25: reset per-request phase state and drop the reference to
            // any previous request's (finished) task when a new request head
            // arrives. Pipelined requests are serialized by NIO's pipeline
            // assistance, so an in-flight request is never mid-generation
            // when a later head is delivered.
            requestPhaseState = RequestPhaseState()
            activeTask = nil
            inFlightRequests += 1
        case .body(var part):
            if body.readableBytes + part.readableBytes > NVMAIHTTPServer.maximumBodyBytes {
                oversized = true
            } else {
                body.writeBuffer(&part)
            }
        case .end:
            guard let head else { return }
            self.head = nil
            // S35: do not retain the (up to 1 MiB) request body buffer across
            // keep-alive requests.
            defer { body = ByteBuffer() }
            if oversized {
                writeError(context, status: .payloadTooLarge,
                           OpenAIErrorEnvelope(message: "request body is too large",
                                               code: "request_too_large"))
                return
            }
            route(head: head, body: body, context: context)
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        // S25: cancel by identity — capture this connection's current task,
        // clear the property, then cancel so a stale reference can never
        // cancel a task that belongs to a later request.
        let task = activeTask
        activeTask = nil
        task?.cancel()
        idleCloseTask?.cancel()
        idleCloseTask = nil
        childChannels.remove(context.channel)
        context.fireChannelInactive()
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        // An I/O error (e.g. a write to a disconnected client) means the
        // stream can no longer be delivered; cancel the generation so it
        // stops promptly, then let the pipeline handle the error.
        activeTask?.cancel()
        context.fireErrorCaught(error)
    }

    private func route(head: HTTPRequestHead,
                       body: ByteBuffer,
                       context: ChannelHandlerContext) {
        // S27: HTTP/1.1 requires a Host header; reject requests without one.
        if head.version == .http1_1, head.headers.first(name: "host") == nil {
            writeError(context, status: .badRequest,
                       OpenAIErrorEnvelope(message: "missing Host header",
                                           code: "missing_host"))
            return
        }
        let path = head.uri.split(separator: "?", maxSplits: 1,
                                  omittingEmptySubsequences: false).first.map(String.init) ?? head.uri
        // A client that sends anthropic-version is speaking the Messages API;
        // the shared paths (/v1/models, 404s) answer in its shape.
        let anthropic = head.headers.first(name: "anthropic-version") != nil
            || path.hasPrefix("/v1/messages")
        let segments = path.split(separator: "/").map(String.init)
        let jsonBody = head.headers.first(name: "content-type")?
            .lowercased().hasPrefix("application/json") == true
        switch (head.method, path) {
        case (.GET, "/health"):
            writeJSON(context, status: .ok, object: ["status": "ok"])
        case (.GET, "/v1/models"):
            if let router {
                writeModelList(router.servedModels, anthropic: anthropic, context: context)
                return
            }
            // Advertise the base model plus the "<model>-fast" alias, which
            // serves the same weights with the CLI-strip heuristic enabled.
            if anthropic {
                writeJSON(context, status: .ok,
                          object: AnthropicBuilder.modelList(ids: [modelID, modelID + "-fast"]),
                          surface: .anthropic)
                return
            }
            let response = OpenAIModelList(
                object: "list",
                data: [
                    .init(id: modelID,
                          object: "model",
                          created: nil,
                          ownedBy: "nvmai"),
                    .init(id: modelID + "-fast",
                          object: "model",
                          created: nil,
                          ownedBy: "nvmai"),
                ])
            writeCodable(context, status: .ok, response)
        case (.HEAD, "/health"), (.HEAD, "/v1/models"):
            // S28: HEAD is answered with headers only.
            writeHeadOnly(context, status: .ok)
        case (.POST, "/v1/chat/completions"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .chat)
                return
            }
            handleCompletion(
                body: body,
                context: context,
                // From the *local* head: `self.head` is cleared when `.end`
                // arrives, before this runs. Reading it here returned nil
                // every time, which is a fix that compiles, ships and does
                // nothing -- caught only because the isolation scenario
                // refuses to run until it sees the header take effect.
                workspace: WorkspaceHeader.value(in: head))
        case (.POST, "/v1/responses"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .responses)
                return
            }
            handleResponses(
                body: body,
                context: context,
                workspace: WorkspaceHeader.value(in: head))
        case (.POST, "/v1/messages"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .anthropic)
                return
            }
            handleMessages(body: body, context: context,
                           workspace: WorkspaceHeader.value(in: head))
        case (.POST, "/v1/messages/count_tokens"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .anthropic)
                return
            }
            handleCountTokens(body: body, context: context)
        case (.POST, "/v1/models/unload"):
            handleUnload(context: context)
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"), (_, "/v1/responses"),
             (_, "/v1/models/unload"), (_, "/v1/messages"), (_, "/v1/messages/count_tokens"):
            writeRequestError(context, .invalid(message: "method not allowed", param: nil,
                                                code: "method_not_allowed"),
                              status: .methodNotAllowed,
                              surface: anthropic ? .anthropic : .chat)
        default:
            if segments.count == 3, segments[0] == "v1", segments[1] == "models" {
                handleModel(id: segments[2], method: head.method, anthropic: anthropic, context: context)
            } else if segments.count >= 3, segments[0] == "v1", segments[1] == "responses" {
                handleStoredResponse(segments: Array(segments.dropFirst(2)),
                                     method: head.method, context: context)
            } else {
                writeRequestError(context, .notFound(message: "route not found", param: nil),
                                  status: .notFound,
                                  surface: anthropic ? .anthropic : .chat)
            }
        }
    }

    private func writeUnsupportedMediaType(_ context: ChannelHandlerContext, surface: APISurface) {
        writeRequestError(context,
                          .invalid(message: "content-type must be application/json",
                                   param: nil, code: "unsupported_media_type"),
                          status: .unsupportedMediaType, surface: surface)
    }

    /// The model a request names. Resolved before validation, so omitted
    /// sampling and the max_tokens bound come from that model rather than
    /// whichever one is resident. Without a router this is the one model,
    /// answered from the backend as before, and the validator still refuses
    /// any other name.
    private func servedModel(named name: String) throws -> ServedModel {
        guard let router else {
            return ServedModel(id: modelID, displayName: modelID,
                               maximumContext: backend.maximumContext,
                               sampling: backend.samplingDefaults,
                               reasoningProfile: reasoningProfile)
        }
        guard let model = router.servedModel(named: name) else {
            throw ServerRequestError.unknownModel
        }
        return model
    }

    /// Validates against `target` and binds the request to it, so the router
    /// loads the model that was validated and the response names it.
    private func validate(_ request: OpenAIChatRequest,
                          for target: ServedModel) throws -> ValidatedChatRequest {
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: target.id, maxContext: target.maximumContext,
            reasoningProfile: target.reasoningProfile,
            sampling: target.sampling)
            .withModel(target.id)
        // Best-effort reasoning: say what was applied when a client asked for
        // a level this model cannot render. Never an error — see
        // `ReasoningFallback`.
        if !validated.reasoningNotes.isEmpty {
            ServerLog.reasoningFallback(id: target.id, notes: validated.reasoningNotes)
        }
        responseModelID = target.id
        return validated
    }

    /// Every catalog model in the shape the client speaks. The "-fast"
    /// aliases are accepted but not listed: doubling a catalog into twice as
    /// many menu entries helps nobody choose between models.
    private func writeModelList(_ models: [ServedModel], anthropic: Bool,
                                context: ChannelHandlerContext) {
        if anthropic {
            writeJSON(context, status: .ok,
                      object: AnthropicBuilder.modelList(
                          models: models.map { (id: $0.id, displayName: $0.displayName) }),
                      surface: .anthropic)
            return
        }
        writeCodable(context, status: .ok, OpenAIModelList(
            object: "list",
            data: models.map {
                .init(id: $0.id, object: "model", created: nil, ownedBy: "nvmai")
            }))
    }

    /// `GET /v1/models/{id}` in either shape.
    private func handleModel(id: String, method: HTTPMethod, anthropic: Bool,
                             context: ChannelHandlerContext) {
        let surface: APISurface = anthropic ? .anthropic : .chat
        guard method == .GET else {
            writeRequestError(context, .invalid(message: "method not allowed", param: nil,
                                                code: "method_not_allowed"),
                              status: .methodNotAllowed, surface: surface)
            return
        }
        let displayName: String
        if let router {
            guard let model = router.servedModel(named: id) else {
                writeRequestError(context, .unknownModel, status: .notFound, surface: surface)
                return
            }
            displayName = model.displayName
        } else {
            guard id == modelID || id == modelID + "-fast" else {
                writeRequestError(context, .unknownModel, status: .notFound, surface: surface)
                return
            }
            displayName = id
        }
        if anthropic {
            writeJSON(context, status: .ok,
                      object: AnthropicBuilder.modelObject(id: id, displayName: displayName),
                      surface: .anthropic)
        } else {
            writeCodable(context, status: .ok,
                         OpenAIModelList.Model(id: id, object: "model", created: nil, ownedBy: "nvmai"))
        }
    }

    /// `GET|DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`,
    /// `GET /v1/responses/{id}/input_items`: the stored side of the
    /// Responses API. Nothing here runs the model.
    private func handleStoredResponse(segments: [String], method: HTTPMethod,
                                      context: ChannelHandlerContext) {
        let id = segments[0]
        let notFound = ServerRequestError.notFound(
            message: "Response with id '\(id)' not found.", param: "id")
        switch (method, segments.count, segments.count > 1 ? segments[1] : "") {
        case (.GET, 1, _):
            guard let entry = responseStore.get(id) else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses); return
            }
            writeData(context, status: .ok, data: entry.responseJSON)
        case (.DELETE, 1, _):
            guard responseStore.delete(id) else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses); return
            }
            writeJSON(context, status: .ok, object: ["id": id, "object": "response", "deleted": true])
        case (.POST, 2, "cancel"):
            guard responseStore.get(id) != nil else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses); return
            }
            // Every response this server produces is foreground and already
            // finished by the time it has an id to cancel.
            writeRequestError(context, .invalid(
                message: "Only responses created with background=true can be cancelled.",
                param: "id", code: "invalid_request"), status: .badRequest, surface: .responses)
        case (.GET, 2, "input_items"):
            guard let entry = responseStore.get(id) else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses); return
            }
            do {
                writeData(context, status: .ok,
                          data: try ResponsesAPIBuilder.inputItemsList(entry.inputItems))
            } catch {
                writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            }
        case (_, 1, _), (_, 2, "cancel"), (_, 2, "input_items"):
            writeRequestError(context, .invalid(message: "method not allowed", param: nil,
                                                code: "method_not_allowed"),
                              status: .methodNotAllowed, surface: .responses)
        default:
            writeRequestError(context, .notFound(message: "route not found", param: nil),
                              status: .notFound, surface: .responses)
        }
    }

    /// Control endpoint: release the model's memory on demand. With residency
    /// managed (--lazy-load / --idle-unload-seconds) this waits for in-flight
    /// requests to drain, then unloads; with a plain session it is a no-op.
    private func handleUnload(context: ChannelHandlerContext) {
        let contextBox = SendableContext(context)
        activeTask = childChannels.startTask {
            // Only a residency-managing backend has anything to release; a
            // plain session reports false without the inference protocol
            // needing to know residency exists.
            let released: Bool
            if let managing = self.backend as? any ResidencyManaging {
                released = await managing.unload()
            } else {
                released = false
            }
            self.writeJSON(contextBox.value, status: .ok,
                           object: ["status": "ok", "unloaded": released])
        }
    }

    private func handleCompletion(body: ByteBuffer,
                                  context: ChannelHandlerContext,
                                  workspace: String? = nil) {
        do {
            // One copy out of the ByteBuffer, not two: a [UInt8] hop would
            // duplicate a body of up to `maximumBodyBytes` before decoding.
            let decoded = try JSONDecoder().decode(
                OpenAIChatRequest.self, from: Data(body.readableBytesView))
            let request = try validate(decoded, for: try servedModel(named: decoded.model))
                .withWorkspace(workspace)
            let responseID = "chatcmpl-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let phaseState = requestPhaseState
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeHeartbeat(contextBox.value)
                      }) else { return }
                let future = self.beginStream(
                    contextBox.value,
                    self.chunk(id: responseID, created: created,
                               delta: ["role": "assistant"],
                               finishReason: nil))
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: responseID, streaming: request.stream)
                let outbox: SSEOutbox? = request.stream
                    ? SSEOutbox(capacity: Self.maximumPendingStreamChunks)
                    : nil
                let drainer = outbox.map { outbox in
                    Task { [self] in
                        await self.drainOutbox(contextBox.value, outbox: outbox)
                    }
                }
                do {
                    let completion = try await self.coordinator.run(onQueued: onQueued) {
                        try Task.checkCancellation()
                        startStream()
                        try await streamState.waitUntilStarted()
                        try Task.checkCancellation()
                        phaseState.set("generating")
                        ServerLog.generating(id: responseID)
                        return try await self.backend.generate(request) { event in
                            guard request.stream, let outbox else { return }
                            self.enqueueChatEvent(event, id: responseID, created: created,
                                                  streamState: streamState,
                                                  outbox: outbox, context: contextBox.value)
                        }
                    }
                    ServerLog.completed(id: responseID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    if request.stream, let outbox {
                        streamState.stop()
                        self.finishStream(contextBox.value,
                                          id: responseID,
                                          created: created,
                                          completion: completion,
                                          includeUsage: request.includeUsage,
                                          outbox: outbox)
                    } else {
                        self.writeCompletion(contextBox.value,
                                             id: responseID,
                                             created: created,
                                             completion: completion)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncFailure(error,
                                            context: contextBox.value,
                                            id: responseID,
                                            phase: phaseState.value,
                                            stream: request.stream,
                                            outbox: outbox,
                                            surface: .chat)
                }
                if let drainer {
                    await Self.awaitDrainer(drainer)
                }
            }
        } catch let error as ServerRequestError {
            writeError(context,
                       status: error == .unknownModel ? .notFound : .badRequest,
                       error.envelope)
        } catch {
            writeError(context, status: .badRequest,
                       OpenAIErrorEnvelope(message: "malformed JSON request",
                                           code: "invalid_json"))
        }
    }

    /// Per-request bookkeeping for a /v1/responses stream: the event sequence
    /// number and the output indices handed to items in the order the model
    /// produced them, so the streamed items and the final object agree.
    /// unchecked-invariant: every field is guarded by `lock`; the sequence is
    /// read from the queued callback (event loop) and the generation task.
    private final class ResponsesStreamState: @unchecked Sendable {
        private let lock = NSLock()
        private var sequence = 0
        private var nextOutput = 0
        private var _messageIndex: Int?
        private var _callIndices: [Int] = []

        func nextSequence() -> Int {
            lock.withLock {
                defer { sequence += 1 }
                return sequence
            }
        }

        var messageIndex: Int? { lock.withLock { _messageIndex } }
        var callIndices: [Int] { lock.withLock { _callIndices } }

        /// The message item's output index, allocated on first use.
        func announceMessage() -> (index: Int, first: Bool) {
            lock.withLock {
                if let index = _messageIndex { return (index, false) }
                let index = nextOutput
                nextOutput += 1
                _messageIndex = index
                return (index, true)
            }
        }

        /// A function-call item's output index and its ordinal among calls.
        func allocateCall() -> (index: Int, ordinal: Int) {
            lock.withLock {
                let index = nextOutput
                nextOutput += 1
                _callIndices.append(index)
                return (index, _callIndices.count - 1)
            }
        }

        /// Every reasoning item so far, by output index, with its whole text:
        /// the done events and the final object both need the full thought.
        private var _reasoning: [(index: Int, text: String)] = []
        private var openReasoning: Int?

        var reasoningItems: [(index: Int, text: String)] { lock.withLock { _reasoning } }

        /// Appends to the open reasoning item, opening one when none is.
        /// The ordinal numbers the item among reasoning items, for its id.
        func appendReasoning(_ text: String) -> (index: Int, ordinal: Int, first: Bool) {
            lock.withLock {
                if let ordinal = openReasoning {
                    _reasoning[ordinal].text += text
                    return (_reasoning[ordinal].index, ordinal, false)
                }
                let index = nextOutput
                nextOutput += 1
                _reasoning.append((index, text))
                openReasoning = _reasoning.count - 1
                return (index, _reasoning.count - 1, true)
            }
        }

        /// Close the open reasoning item, returning it if there was one.
        func closeReasoning() -> (index: Int, ordinal: Int, text: String)? {
            lock.withLock {
                guard let ordinal = openReasoning else { return nil }
                openReasoning = nil
                return (_reasoning[ordinal].index, ordinal, _reasoning[ordinal].text)
            }
        }
    }

    private static func eventFrame(name: String, object: [String: Any]) -> Data? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return nil
        }
        return Self.sseFrame("event: " + name + "\ndata: " + String(decoding: data, as: UTF8.self))
    }

    /// The frames that end a stream after a failure, in the surface's shape:
    /// chat sends an error object then [DONE]; the Responses API and the
    /// Messages API send a typed `error` event and no terminator.
    private static func failureFrames(_ envelope: OpenAIErrorEnvelope,
                                      surface: APISurface,
                                      requestID: String? = nil) -> [Data] {
        switch surface {
        case .chat:
            return errorFrame(envelope).map { [$0, doneFrame()] } ?? [doneFrame()]
        case .responses:
            let object: [String: Any] = [
                "type": "error", "code": envelope.error.code,
                "message": envelope.error.message,
                "param": envelope.error.param.map { $0 as Any } ?? NSNull(),
            ]
            return eventFrame(name: "error", object: object).map { [$0] } ?? []
        case .anthropic:
            let detail = AnthropicErrorEnvelope(
                type: envelope.error.type == "server_error" ? "api_error" : envelope.error.type,
                message: envelope.error.message, requestID: requestID)
            guard let data = try? JSONEncoder().encode(detail) else { return [] }
            return [sseFrame("event: error\ndata: " + String(decoding: data, as: UTF8.self))]
        }
    }

    /// An `item_reference` names an output item of a stored response; the
    /// item itself takes its place in the input. Unknown ids stay as they
    /// are and the mapper refuses them by id.
    private func resolveReferences(_ items: [ResponsesAPIRequest.Item]) -> [ResponsesAPIRequest.Item] {
        items.map { item in
            guard item.resolvedType == "item_reference", let id = item.id,
                  let stored = responseStore.item(withID: id) else { return item }
            return stored
        }
    }

    /// The API reports a failed generation as a response object in state
    /// "failed" carrying the error, then ends the stream.
    private func responsesFailureFrames(
        id: String, created: Int, echo: ResponsesAPIEcho, itemState: ResponsesStreamState
    ) -> @Sendable (OpenAIErrorEnvelope) -> [Data] {
        { envelope in
            let failed = ResponsesAPIBuilder.responseObject(
                id: id, created: created, model: self.responseModelID,
                status: "failed", output: [], usage: nil, echo: echo,
                error: (envelope.error.code, envelope.error.message))
            let event = ResponsesAPIBuilder.event(
                "response.failed", sequence: itemState.nextSequence(), ["response": failed])
            return Self.eventFrame(name: "response.failed", object: event).map { [$0] } ?? []
        }
    }

    /// The stored conversation `previous_response_id` continues, or none.
    private func priorConversation(
        _ request: ResponsesAPIRequest
    ) throws -> [ResponsesAPIRequest.Item] {
        guard let previous = request.previousResponseID else { return [] }
        guard let entry = responseStore.get(previous) else {
            throw ServerRequestError.notFound(
                message: "Previous response with id '\(previous)' not found.",
                param: "previous_response_id")
        }
        return entry.conversation
    }

    /// OpenAI Responses API endpoint (`POST /v1/responses`). The request is
    /// mapped onto the chat-completions path (see ResponsesAPIMapper) and the
    /// generation is streamed back as Responses-API SSE events (or returned
    /// as a single response object when stream is false). A finished response
    /// is stored (unless store=false) so a later request can continue it by
    /// previous_response_id and the retrieval endpoints can serve it.
    private func handleResponses(body: ByteBuffer,
                                 context: ChannelHandlerContext,
                                 workspace: String? = nil) {
        do {
            let decoded = try JSONDecoder().decode(
                ResponsesAPIRequest.self, from: Data(body.readableBytesView))
            let target = try servedModel(named: decoded.model)
            let prior = try priorConversation(decoded)
            let inputItems = resolveReferences(decoded.inputItems)
            let chatRequest = try ResponsesAPIMapper.chatRequest(
                decoded, priorItems: prior, inputItems: inputItems)
            let request = try validate(chatRequest, for: target)
                .withWorkspace(workspace)
            let echo = ResponsesAPIEcho(request: decoded,
                                        effectiveEffort: target.reasoningProfile.effectiveEffort)
            let responseID = ResponsesAPIBuilder.responseID()
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let itemState = ResponsesStreamState()
            let phaseState = requestPhaseState
            let storedInput = prior + inputItems
            let stores = decoded.stores
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeHeartbeat(contextBox.value)
                      }) else { return }
                let future = self.beginResponsesStream(
                    contextBox.value, id: responseID, created: created,
                    echo: echo, itemState: itemState)
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: responseID, streaming: request.stream)
                let outbox: SSEOutbox? = request.stream
                    ? SSEOutbox(capacity: Self.maximumPendingStreamChunks)
                    : nil
                let drainer = outbox.map { outbox in
                    Task { [self] in
                        await self.drainOutbox(contextBox.value, outbox: outbox)
                    }
                }
                do {
                    let completion = try await self.coordinator.run(onQueued: onQueued) {
                        try Task.checkCancellation()
                        startStream()
                        try await streamState.waitUntilStarted()
                        try Task.checkCancellation()
                        phaseState.set("generating")
                        ServerLog.generating(id: responseID)
                        return try await self.backend.generate(request) { event in
                            guard request.stream, let outbox else { return }
                            self.enqueueResponsesEvent(event, id: responseID, echo: echo,
                                                       itemState: itemState,
                                                       outbox: outbox, context: contextBox.value)
                        }
                    }
                    ServerLog.completed(id: responseID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    let final: [String: Any]
                    if request.stream, let outbox {
                        streamState.stop()
                        final = self.finishResponsesStream(
                            contextBox.value, id: responseID, created: created, echo: echo,
                            completion: completion, itemState: itemState, outbox: outbox)
                    } else {
                        final = self.finalResponsesObject(
                            id: responseID, created: created, echo: echo,
                            completion: completion, itemState: nil)
                        self.writeJSON(contextBox.value, status: .ok, object: final)
                    }
                    if stores {
                        self.storeResponse(id: responseID, object: final,
                                           input: storedInput, completion: completion,
                                           namespaces: echo.namespaces)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncFailure(
                        error, context: contextBox.value, id: responseID,
                        phase: phaseState.value, stream: request.stream, outbox: outbox,
                        surface: .responses,
                        failureFrames: self.responsesFailureFrames(
                            id: responseID, created: created, echo: echo, itemState: itemState))
                }
                if let drainer {
                    await Self.awaitDrainer(drainer)
                }
            }
        } catch let error as ServerRequestError {
            writeRequestError(context, error,
                              status: HTTPResponseStatus(statusCode: error.httpStatus),
                              surface: .responses)
        } catch {
            writeRequestError(context, .invalid(message: "malformed JSON request",
                                                param: nil, code: "invalid_json"),
                              status: .badRequest, surface: .responses)
        }
    }

    /// The response object of a finished generation. In a stream the output
    /// order is the order items were announced; otherwise message first.
    private func finalResponsesObject(id: String,
                                      created: Int,
                                      echo: ResponsesAPIEcho,
                                      completion: ServerCompletion,
                                      itemState: ResponsesStreamState?) -> [String: Any] {
        let ids = ResponsesAPIBuilder.itemIDs(responseID: id, completion: completion)
        var output: [[String: Any]]
        if let itemState {
            var slots: [Int: [String: Any]] = [:]
            for (ordinal, item) in itemState.reasoningItems.enumerated() {
                slots[item.index] = ResponsesAPIBuilder.reasoningItem(
                    id: ResponsesAPIBuilder.reasoningItemID(responseID: id, index: ordinal),
                    text: item.text)
            }
            if let index = itemState.messageIndex {
                slots[index] = ResponsesAPIBuilder.messageItem(
                    id: ids.message, role: "assistant", text: completion.content, status: "completed")
            }
            for (ordinal, index) in itemState.callIndices.enumerated()
            where ordinal < completion.toolCalls.count {
                let call = completion.toolCalls[ordinal]
                slots[index] = ResponsesAPIBuilder.functionCallItem(
                    id: ids.calls[ordinal], name: call.name, arguments: call.argumentsJSON,
                    callID: call.id, status: "completed", namespace: echo.namespaces[call.name])
            }
            output = slots.keys.sorted().compactMap { slots[$0] }
        } else {
            output = ResponsesAPIBuilder.outputItems(completion: completion, responseID: id,
                                                     namespaces: echo.namespaces)
        }
        if output.isEmpty {
            output = [ResponsesAPIBuilder.messageItem(
                id: ids.message, role: "assistant", text: "", status: "completed")]
        }
        let terminal = ResponsesAPIBuilder.terminalStatus(for: completion)
        return ResponsesAPIBuilder.responseObject(
            id: id, created: created, model: responseModelID, status: terminal.status,
            output: output, usage: completion.usage, echo: echo,
            incompleteReason: terminal.reason,
            completedAt: Int(Date().timeIntervalSince1970))
    }

    private func storeResponse(id: String,
                               object: [String: Any],
                               input: [ResponsesAPIRequest.Item],
                               completion: ServerCompletion,
                               namespaces: [String: String]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        responseStore.put(id: id, entry: ResponseStore.Entry(
            responseJSON: data, inputItems: input,
            outputItems: ResponsesAPIMapper.outputAsInput(
                completion: completion, responseID: id, namespaces: namespaces),
            created: Date()))
    }

    private func beginResponsesStream(_ context: ChannelHandlerContext,
                                      id: String,
                                      created: Int,
                                      echo: ResponsesAPIEcho,
                                      itemState: ResponsesStreamState) -> EventLoopFuture<Void> {
        let response = ResponsesAPIBuilder.responseObject(
            id: id, created: created, model: responseModelID, status: "in_progress",
            output: [], usage: nil, echo: echo)
        var frames = Data()
        for name in ["response.created", "response.in_progress"] {
            let event = ResponsesAPIBuilder.event(name, sequence: itemState.nextSequence(),
                                                  ["response": response])
            if let frame = Self.eventFrame(name: name, object: event) {
                frames.append(frame)
            }
        }
        return writeStreamHead(context, initialFrames: frames, extraHeaders: [])
    }

    /// Enqueue one Responses-API event, numbering it in stream order.
    private func responsesEvent(_ type: String,
                                _ fields: [String: Any],
                                itemState: ResponsesStreamState,
                                outbox: SSEOutbox,
                                context: ChannelHandlerContext) {
        let object = ResponsesAPIBuilder.event(type, sequence: itemState.nextSequence(), fields)
        guard let frame = Self.eventFrame(name: type, object: object) else {
            failStream(outbox: outbox, context: context,
                       message: "stream response could not be encoded",
                       code: "internal_error", surface: .responses)
            return
        }
        guard outbox.enqueue(frame) else {
            failStream(outbox: outbox, context: context,
                       message: "stream backpressure limit exceeded; client is too slow",
                       code: "stream_overflow", surface: .responses)
            return
        }
    }

    private func enqueueResponsesEvent(_ event: ServerInferenceEvent,
                                       id: String,
                                       echo: ResponsesAPIEcho,
                                       itemState: ResponsesStreamState,
                                       outbox: SSEOutbox,
                                       context: ChannelHandlerContext) {
        switch event {
        case .content(let text):
            enqueueResponsesContentDelta(id: id, text: text, itemState: itemState,
                                         outbox: outbox, context: context)
        case .reasoning(let text):
            enqueueResponsesReasoningDelta(id: id, text: text, itemState: itemState,
                                           outbox: outbox, context: context)
        case .toolCall(let call):
            enqueueResponsesToolCall(id: id, call: call, itemState: itemState,
                                     namespace: echo.namespaces[call.name],
                                     outbox: outbox, context: context)
        }
    }

    /// Thoughts stream as a reasoning item's summary text, ahead of the
    /// message, the way the API's own reasoning models order them.
    private func enqueueResponsesReasoningDelta(id: String,
                                                text: String,
                                                itemState: ResponsesStreamState,
                                                outbox: SSEOutbox,
                                                context: ChannelHandlerContext) {
        let (index, ordinal, first) = itemState.appendReasoning(text)
        let itemID = ResponsesAPIBuilder.reasoningItemID(responseID: id, index: ordinal)
        if first {
            responsesEvent("response.output_item.added",
                           ["output_index": index,
                            "item": ["id": itemID, "type": "reasoning", "summary": []]],
                           itemState: itemState, outbox: outbox, context: context)
            responsesEvent("response.reasoning_summary_part.added",
                           ["item_id": itemID, "output_index": index, "summary_index": 0,
                            "part": ResponsesAPIBuilder.summaryTextPart("")],
                           itemState: itemState, outbox: outbox, context: context)
        }
        responsesEvent("response.reasoning_summary_text.delta",
                       ["item_id": itemID, "output_index": index, "summary_index": 0,
                        "delta": text],
                       itemState: itemState, outbox: outbox, context: context)
    }

    /// Finish the open reasoning item, if any: the model has moved on to its
    /// answer or a call, and a client renders the thought as complete.
    private func closeResponsesReasoning(id: String,
                                         itemState: ResponsesStreamState,
                                         outbox: SSEOutbox,
                                         context: ChannelHandlerContext) {
        guard let (index, ordinal, text) = itemState.closeReasoning() else { return }
        let itemID = ResponsesAPIBuilder.reasoningItemID(responseID: id, index: ordinal)
        responsesEvent("response.reasoning_summary_text.done",
                       ["item_id": itemID, "output_index": index, "summary_index": 0,
                        "text": text],
                       itemState: itemState, outbox: outbox, context: context)
        responsesEvent("response.reasoning_summary_part.done",
                       ["item_id": itemID, "output_index": index, "summary_index": 0,
                        "part": ResponsesAPIBuilder.summaryTextPart(text)],
                       itemState: itemState, outbox: outbox, context: context)
        responsesEvent("response.output_item.done",
                       ["output_index": index,
                        "item": ResponsesAPIBuilder.reasoningItem(id: itemID, text: text)],
                       itemState: itemState, outbox: outbox, context: context)
    }

    private func enqueueResponsesContentDelta(id: String,
                                              text: String,
                                              itemState: ResponsesStreamState,
                                              outbox: SSEOutbox,
                                              context: ChannelHandlerContext) {
        closeResponsesReasoning(id: id, itemState: itemState, outbox: outbox, context: context)
        let itemID = ResponsesAPIBuilder.messageItemID(responseID: id)
        let (index, first) = itemState.announceMessage()
        if first {
            responsesEvent("response.output_item.added",
                           ["output_index": index,
                            "item": ["id": itemID, "type": "message", "role": "assistant",
                                     "status": "in_progress", "content": []]],
                           itemState: itemState, outbox: outbox, context: context)
            responsesEvent("response.content_part.added",
                           ["item_id": itemID, "output_index": index, "content_index": 0,
                            "part": ResponsesAPIBuilder.outputTextPart("")],
                           itemState: itemState, outbox: outbox, context: context)
        }
        responsesEvent("response.output_text.delta",
                       ["item_id": itemID, "output_index": index, "content_index": 0,
                        "delta": text, "logprobs": []],
                       itemState: itemState, outbox: outbox, context: context)
    }

    /// A tool call arrives complete from the decoder, so its whole item life
    /// cycle is streamed at once: added, argument deltas, done, item done.
    private func enqueueResponsesToolCall(id: String,
                                          call: ParsedToolCall,
                                          itemState: ResponsesStreamState,
                                          namespace: String?,
                                          outbox: SSEOutbox,
                                          context: ChannelHandlerContext) {
        closeResponsesReasoning(id: id, itemState: itemState, outbox: outbox, context: context)
        let (index, ordinal) = itemState.allocateCall()
        let itemID = ResponsesAPIBuilder.functionCallItemID(responseID: id, index: ordinal)
        responsesEvent("response.output_item.added",
                       ["output_index": index,
                        "item": ResponsesAPIBuilder.functionCallItem(
                            id: itemID, name: call.name, arguments: "", callID: call.id,
                            status: "in_progress", namespace: namespace)],
                       itemState: itemState, outbox: outbox, context: context)
        for fragment in utf8Fragments(call.argumentsJSON, maximumBytes: 1024) {
            responsesEvent("response.function_call_arguments.delta",
                           ["item_id": itemID, "output_index": index, "delta": fragment],
                           itemState: itemState, outbox: outbox, context: context)
        }
        responsesEvent("response.function_call_arguments.done",
                       ["item_id": itemID, "output_index": index, "name": call.name,
                        "arguments": call.argumentsJSON],
                       itemState: itemState, outbox: outbox, context: context)
        responsesEvent("response.output_item.done",
                       ["output_index": index,
                        "item": ResponsesAPIBuilder.functionCallItem(
                            id: itemID, name: call.name, arguments: call.argumentsJSON,
                            callID: call.id, status: "completed", namespace: namespace)],
                       itemState: itemState, outbox: outbox, context: context)
    }

    /// Close the message item, then end the stream with the terminal object:
    /// `response.completed`, or `response.incomplete` when the output cap
    /// cut the generation. Returns the object so it can be stored.
    private func finishResponsesStream(_ context: ChannelHandlerContext,
                                       id: String,
                                       created: Int,
                                       echo: ResponsesAPIEcho,
                                       completion: ServerCompletion,
                                       itemState: ResponsesStreamState,
                                       outbox: SSEOutbox) -> [String: Any] {
        let itemID = ResponsesAPIBuilder.messageItemID(responseID: id)
        // A turn cut off mid-thought still finishes its reasoning item.
        closeResponsesReasoning(id: id, itemState: itemState, outbox: outbox, context: context)
        // A turn with no text and no calls still has one (empty) message
        // item, as the API's own output does.
        if itemState.messageIndex == nil, completion.toolCalls.isEmpty {
            enqueueResponsesContentDelta(id: id, text: "", itemState: itemState,
                                         outbox: outbox, context: context)
        }
        if let index = itemState.messageIndex {
            responsesEvent("response.output_text.done",
                           ["item_id": itemID, "output_index": index, "content_index": 0,
                            "text": completion.content, "logprobs": []],
                           itemState: itemState, outbox: outbox, context: context)
            responsesEvent("response.content_part.done",
                           ["item_id": itemID, "output_index": index, "content_index": 0,
                            "part": ResponsesAPIBuilder.outputTextPart(completion.content)],
                           itemState: itemState, outbox: outbox, context: context)
            responsesEvent("response.output_item.done",
                           ["output_index": index,
                            "item": ResponsesAPIBuilder.messageItem(
                                id: itemID, role: "assistant",
                                text: completion.content, status: "completed")],
                           itemState: itemState, outbox: outbox, context: context)
        }
        let final = finalResponsesObject(id: id, created: created, echo: echo,
                                         completion: completion, itemState: itemState)
        let name = (final["status"] as? String) == "incomplete"
            ? "response.incomplete" : "response.completed"
        responsesEvent(name, ["response": final],
                       itemState: itemState, outbox: outbox, context: context)
        // The Responses API has no [DONE] terminator; the final event is it.
        outbox.enqueueTerminal([], closeWhenDrained: false)
        return final
    }

    // MARK: Anthropic Messages API

    /// Content-block bookkeeping for one /v1/messages stream.
    /// unchecked-invariant: guarded by `lock`, for the same reason as the
    /// Responses state above.
    private final class AnthropicStreamState: @unchecked Sendable {
        /// The two kinds of block that stream as deltas. A tool_use block
        /// arrives whole and is never left open.
        enum Kind {
            case text
            case thinking
        }

        struct Block {
            let index: Int
            let kind: Kind
        }

        private let lock = NSLock()
        private var nextIndex = 0
        private var open: Block?
        private var _announced = false

        /// True once a text or tool_use block has been opened. Thinking does
        /// not count: a message whose only block is a thought still gets its
        /// (empty) text block, as the non-streamed content does.
        var announced: Bool { lock.withLock { _announced } }

        /// The open block of `kind`, opening one when none is. Returns the
        /// block of the other kind it had to close first, for the caller to
        /// stop, since a thought and the answer never share a block.
        func block(_ kind: Kind) -> (index: Int, first: Bool, closed: Block?) {
            lock.withLock {
                if kind == .text { _announced = true }
                if let block = open, block.kind == kind { return (block.index, false, nil) }
                let closed = open
                let index = nextIndex
                nextIndex += 1
                open = Block(index: index, kind: kind)
                return (index, true, closed)
            }
        }

        /// Close the open block, returning it if there was one.
        func close() -> Block? {
            lock.withLock {
                defer { open = nil }
                return open
            }
        }

        func allocate() -> Int {
            lock.withLock {
                _announced = true
                defer { nextIndex += 1 }
                return nextIndex
            }
        }
    }

    private static func anthropicFrame(_ object: [String: Any]) -> Data? {
        guard let type = object["type"] as? String else { return nil }
        return eventFrame(name: type, object: object)
    }

    /// Anthropic Messages API endpoint (`POST /v1/messages`). Mapped onto the
    /// same validated chat request as the OpenAI paths; answered in the
    /// Messages API's own object and event shapes.
    private func handleMessages(body: ByteBuffer,
                                context: ChannelHandlerContext,
                                workspace: String? = nil) {
        let requestID = AnthropicBuilder.requestID()
        do {
            let decoded = try JSONDecoder().decode(
                AnthropicMessagesRequest.self, from: Data(body.readableBytesView))
            let target = try servedModel(named: decoded.model)
            let chatRequest = try AnthropicMapper.chatRequest(
                decoded, profile: target.reasoningProfile, maxContext: target.maximumContext)
            let request = try validate(chatRequest, for: target)
                .withWorkspace(workspace)
            let messageID = AnthropicBuilder.messageID()
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let blockState = AnthropicStreamState()
            let phaseState = requestPhaseState
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeAnthropicPing(contextBox.value)
                      }) else { return }
                let future = self.beginAnthropicStream(
                    contextBox.value, id: messageID, requestID: requestID)
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: messageID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: messageID, streaming: request.stream)
                let outbox: SSEOutbox? = request.stream
                    ? SSEOutbox(capacity: Self.maximumPendingStreamChunks)
                    : nil
                let drainer = outbox.map { outbox in
                    Task { [self] in
                        await self.drainOutbox(contextBox.value, outbox: outbox)
                    }
                }
                do {
                    let completion = try await self.coordinator.run(onQueued: onQueued) {
                        try Task.checkCancellation()
                        startStream()
                        try await streamState.waitUntilStarted()
                        try Task.checkCancellation()
                        phaseState.set("generating")
                        ServerLog.generating(id: messageID)
                        return try await self.backend.generate(request) { event in
                            guard request.stream, let outbox else { return }
                            switch event {
                            case .content(let text):
                                self.enqueueAnthropicDelta(
                                    .text, text, blockState: blockState,
                                    outbox: outbox, context: contextBox.value)
                            case .reasoning(let text):
                                self.enqueueAnthropicDelta(
                                    .thinking, text, blockState: blockState,
                                    outbox: outbox, context: contextBox.value)
                            case .toolCall(let call):
                                self.enqueueAnthropicToolUse(
                                    call, blockState: blockState,
                                    outbox: outbox, context: contextBox.value)
                            }
                        }
                    }
                    ServerLog.completed(id: messageID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    if request.stream, let outbox {
                        streamState.stop()
                        self.finishAnthropicStream(
                            contextBox.value, completion: completion,
                            blockState: blockState, outbox: outbox)
                    } else {
                        let stop = AnthropicBuilder.stopReason(for: completion)
                        self.writeJSON(
                            contextBox.value, status: .ok,
                            object: AnthropicBuilder.messageObject(
                                id: messageID, model: self.responseModelID,
                                content: AnthropicBuilder.contentBlocks(completion),
                                stopReason: stop.reason, stopSequence: stop.sequence,
                                usage: AnthropicBuilder.usageObject(completion.usage)),
                            surface: .anthropic, requestID: requestID)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncFailure(
                        error, context: contextBox.value, id: messageID,
                        phase: phaseState.value, stream: request.stream, outbox: outbox,
                        surface: .anthropic, requestID: requestID)
                }
                if let drainer {
                    await Self.awaitDrainer(drainer)
                }
            }
        } catch let error as ServerRequestError {
            writeRequestError(context, error,
                              status: HTTPResponseStatus(statusCode: error.httpStatus),
                              surface: .anthropic, requestID: requestID)
        } catch {
            writeRequestError(context, .invalid(message: "malformed JSON request",
                                                param: nil, code: "invalid_json"),
                              status: .badRequest, surface: .anthropic, requestID: requestID)
        }
    }

    /// `POST /v1/messages/count_tokens`: the prompt tokens the request would
    /// occupy, from the backend's own tokenizer. 501 when the backend has
    /// none to count with.
    private func handleCountTokens(body: ByteBuffer,
                                   context: ChannelHandlerContext) {
        let requestID = AnthropicBuilder.requestID()
        do {
            let decoded = try JSONDecoder().decode(
                AnthropicCountTokensRequest.self, from: Data(body.readableBytesView))
            let target = try servedModel(named: decoded.model)
            let chatRequest = try AnthropicMapper.chatRequest(counting: decoded,
                                                              profile: target.reasoningProfile)
            let request = try validate(chatRequest, for: target)
            guard let counting = backend as? any PromptTokenCounting else {
                throw ServerRequestError.unsupportedOperation("count_tokens")
            }
            let contextBox = SendableContext(context)
            activeTask = childChannels.startTask {
                do {
                    let count = try await counting.countPromptTokens(request)
                    self.writeJSON(contextBox.value, status: .ok, object: ["input_tokens": count],
                                   surface: .anthropic, requestID: requestID)
                } catch let error as ServerRequestError {
                    self.writeRequestError(contextBox.value, error,
                                           status: HTTPResponseStatus(statusCode: error.httpStatus),
                                           surface: .anthropic, requestID: requestID)
                } catch {
                    ServerLog.failed(id: requestID, phase: "counting", status: 500, error: error)
                    self.writeCodable(contextBox.value, status: .internalServerError,
                                      AnthropicErrorEnvelope(type: "api_error",
                                                             message: "token counting failed",
                                                             requestID: requestID),
                                      extraHeaders: [("request-id", requestID)])
                }
            }
        } catch let error as ServerRequestError {
            writeRequestError(context, error,
                              status: HTTPResponseStatus(statusCode: error.httpStatus),
                              surface: .anthropic, requestID: requestID)
        } catch {
            writeRequestError(context, .invalid(message: "malformed JSON request",
                                                param: nil, code: "invalid_json"),
                              status: .badRequest, surface: .anthropic, requestID: requestID)
        }
    }

    private func beginAnthropicStream(_ context: ChannelHandlerContext,
                                      id: String,
                                      requestID: String) -> EventLoopFuture<Void> {
        let message = AnthropicBuilder.messageObject(
            id: id, model: responseModelID, content: [], stopReason: nil, stopSequence: nil,
            usage: ["input_tokens": 0, "output_tokens": 0,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0])
        let frame = Self.anthropicFrame(["type": "message_start", "message": message]) ?? Data()
        return writeStreamHead(context, initialFrames: frame, extraHeaders: [("request-id", requestID)])
    }

    private func anthropicEvent(_ object: [String: Any],
                                outbox: SSEOutbox,
                                context: ChannelHandlerContext) {
        guard let frame = Self.anthropicFrame(object) else {
            failStream(outbox: outbox, context: context,
                       message: "stream response could not be encoded",
                       code: "internal_error", surface: .anthropic)
            return
        }
        guard outbox.enqueue(frame) else {
            failStream(outbox: outbox, context: context,
                       message: "stream backpressure limit exceeded; client is too slow",
                       code: "stream_overflow", surface: .anthropic)
            return
        }
    }

    /// Text or thinking, into the open block of that kind. A thought streams
    /// as its own `thinking` block ahead of the text, with `thinking_delta`s,
    /// as the Messages API streams extended thinking.
    private func enqueueAnthropicDelta(_ kind: AnthropicStreamState.Kind,
                                       _ text: String,
                                       blockState: AnthropicStreamState,
                                       outbox: SSEOutbox,
                                       context: ChannelHandlerContext) {
        let (index, first, closed) = blockState.block(kind)
        if let closed {
            stopAnthropicBlock(closed, outbox: outbox, context: context)
        }
        if first {
            let start: [String: Any] = kind == .text
                ? ["type": "text", "text": ""]
                : ["type": "thinking", "thinking": "",
                   "signature": AnthropicBuilder.thinkingSignature]
            anthropicEvent(["type": "content_block_start", "index": index,
                            "content_block": start],
                           outbox: outbox, context: context)
        }
        let delta: [String: Any] = kind == .text
            ? ["type": "text_delta", "text": text]
            : ["type": "thinking_delta", "thinking": text]
        anthropicEvent(["type": "content_block_delta", "index": index, "delta": delta],
                       outbox: outbox, context: context)
    }

    /// End a streamed block. A thinking block is signed first, as the API
    /// always does just before its stop, so a client that assembles the
    /// block from its deltas ends up with the same object as the
    /// non-streamed content (see `AnthropicBuilder.thinkingSignature`).
    private func stopAnthropicBlock(_ block: AnthropicStreamState.Block,
                                    outbox: SSEOutbox,
                                    context: ChannelHandlerContext) {
        if block.kind == .thinking {
            anthropicEvent(["type": "content_block_delta", "index": block.index,
                            "delta": ["type": "signature_delta",
                                      "signature": AnthropicBuilder.thinkingSignature]],
                           outbox: outbox, context: context)
        }
        anthropicEvent(["type": "content_block_stop", "index": block.index],
                       outbox: outbox, context: context)
    }

    private func enqueueAnthropicToolUse(_ call: ParsedToolCall,
                                         blockState: AnthropicStreamState,
                                         outbox: SSEOutbox,
                                         context: ChannelHandlerContext) {
        if let open = blockState.close() {
            stopAnthropicBlock(open, outbox: outbox, context: context)
        }
        let index = blockState.allocate()
        anthropicEvent(["type": "content_block_start", "index": index,
                        "content_block": ["type": "tool_use", "id": call.id,
                                          "name": call.name, "input": [:]]],
                       outbox: outbox, context: context)
        for fragment in utf8Fragments(call.argumentsJSON, maximumBytes: 1024) {
            anthropicEvent(["type": "content_block_delta", "index": index,
                            "delta": ["type": "input_json_delta", "partial_json": fragment]],
                           outbox: outbox, context: context)
        }
        anthropicEvent(["type": "content_block_stop", "index": index],
                       outbox: outbox, context: context)
    }

    private func finishAnthropicStream(_ context: ChannelHandlerContext,
                                       completion: ServerCompletion,
                                       blockState: AnthropicStreamState,
                                       outbox: SSEOutbox) {
        // A message always carries at least one text or tool_use block.
        if !blockState.announced {
            enqueueAnthropicDelta(.text, "", blockState: blockState, outbox: outbox,
                                  context: context)
        }
        if let open = blockState.close() {
            stopAnthropicBlock(open, outbox: outbox, context: context)
        }
        let stop = AnthropicBuilder.stopReason(for: completion)
        anthropicEvent(["type": "message_delta",
                        "delta": ["stop_reason": stop.reason,
                                  "stop_sequence": stop.sequence.map { $0 as Any } ?? NSNull()],
                        "usage": AnthropicBuilder.usageObject(completion.usage)],
                       outbox: outbox, context: context)
        anthropicEvent(["type": "message_stop"], outbox: outbox, context: context)
        outbox.enqueueTerminal([], closeWhenDrained: false)
    }

    private func writeAnthropicPing(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: "event: ping\ndata: {\"type\": \"ping\"}\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    // MARK: Shared response plumbing

    /// Start an SSE response: the head, then whatever frames the surface
    /// opens with.
    private func writeStreamHead(_ context: ChannelHandlerContext,
                                 initialFrames: Data,
                                 extraHeaders: [(String, String)]) -> EventLoopFuture<Void> {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
        for (name, value) in extraHeaders {
            headers.add(name: name, value: value)
        }
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)), promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: initialFrames.count)
            buffer.writeBytes(initialFrames)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    /// A request error in the surface's envelope, with the status the error
    /// maps to.
    private func writeRequestError(_ context: ChannelHandlerContext,
                                   _ error: ServerRequestError,
                                   status: HTTPResponseStatus,
                                   surface: APISurface,
                                   requestID: String? = nil) {
        switch surface {
        case .chat, .responses:
            writeCodable(context, status: status, error.envelope)
        case .anthropic:
            let id = requestID ?? AnthropicBuilder.requestID()
            writeCodable(context, status: status,
                         AnthropicErrorEnvelope.from(error, requestID: id),
                         extraHeaders: [("request-id", id)])
        }
    }

    private func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any,
                           surface: APISurface,
                           requestID: String? = nil) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        let headers: [(String, String)] = surface == .anthropic
            ? [("request-id", requestID ?? AnthropicBuilder.requestID())] : []
        writeData(context, status: status, data: data, extraHeaders: headers)
    }

    /// A generation that failed after the request was accepted. Streams get
    /// the surface's failure frames and are closed; a cancelled stream (the
    /// client left, or shutdown) just ends. Non-streaming requests get the
    /// error envelope with its status.
    private func handleAsyncFailure(_ error: Error,
                                    context: ChannelHandlerContext,
                                    id: String,
                                    phase: String,
                                    stream: Bool,
                                    outbox: SSEOutbox?,
                                    surface: APISurface,
                                    requestID: String? = nil,
                                    failureFrames: (@Sendable (OpenAIErrorEnvelope) -> [Data])? = nil) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            status = HTTPResponseStatus(statusCode: requestError.httpStatus)
            envelope = requestError.envelope
        } else {
            status = .internalServerError
            envelope = OpenAIErrorEnvelope(
                message: "generation failed; see NVMAIServer stderr",
                code: "internal_error",
                type: "server_error")
        }
        if !(error is CancellationError) {
            ServerLog.failed(id: id, phase: phase, status: status.code, error: error)
        }
        if stream, let outbox {
            // S5/S20: never leave a streaming client without a terminal frame.
            if error is CancellationError {
                outbox.enqueueTerminal(surface == .chat ? [Self.doneFrame()] : [],
                                       closeWhenDrained: true)
            } else {
                let frames = failureFrames?(envelope)
                    ?? Self.failureFrames(envelope, surface: surface, requestID: requestID)
                outbox.enqueueTerminal(frames, closeWhenDrained: true)
            }
            return
        }
        if error is CancellationError {
            // S20: the client disconnected or the server is shutting down;
            // there is no one to write to. Do not emit a misleading 500.
            return
        }
        if let requestError = error as? ServerRequestError {
            writeRequestError(context, requestError, status: status, surface: surface,
                              requestID: requestID)
        } else if surface == .anthropic {
            let id = requestID ?? AnthropicBuilder.requestID()
            writeCodable(context, status: status,
                         AnthropicErrorEnvelope(type: "api_error", message: envelope.error.message,
                                                requestID: id),
                         extraHeaders: [("request-id", id)])
        } else {
            writeError(context, status: status, envelope)
        }
    }

    /// One streamed chat event as its chunk. Reasoning rides in
    /// `delta.reasoning_content`, the vLLM and DeepSeek convention that
    /// Qwen Code, OpenCode and their kind read; a client that knows no
    /// such field ignores it and sees the answer alone.
    private func enqueueChatEvent(_ event: ServerInferenceEvent,
                                  id: String,
                                  created: Int,
                                  streamState: StreamState,
                                  outbox: SSEOutbox,
                                  context: ChannelHandlerContext) {
        switch event {
        case .content(let text):
            enqueueStreamChunk(
                chunk(id: id, created: created, delta: ["content": text], finishReason: nil),
                outbox: outbox, context: context)
        case .reasoning(let text):
            enqueueStreamChunk(
                chunk(id: id, created: created, delta: ["reasoning_content": text],
                      finishReason: nil),
                outbox: outbox, context: context)
        case .toolCall(let call):
            enqueueToolCallChunks(id: id, created: created,
                                  toolIndex: streamState.nextToolIndex(), call: call,
                                  outbox: outbox, context: context)
        }
    }

    private func writeCompletion(_ context: ChannelHandlerContext,
                                 id: String,
                                 created: Int,
                                 completion: ServerCompletion) {
        let encodedContent: Any =
            completion.content.isEmpty && !completion.toolCalls.isEmpty
                ? NSNull()
                : completion.content
        var message: [String: Any] = [
            "role": "assistant",
            "content": encodedContent,
        ]
        // Absent rather than empty with thinking off, so that response is
        // byte for byte what it was.
        if !completion.reasoning.isEmpty {
            message["reasoning_content"] = completion.reasoning
        }
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map(toolCallObject)
        }
        let object: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": responseModelID,
            "choices": [[
                "index": 0,
                "message": message,
                "finish_reason": completion.finishReason,
            ]],
            "usage": usageObject(completion.usage),
        ]
        writeJSON(context, status: .ok, object: object)
    }

    private func beginStream(
        _ context: ChannelHandlerContext,
        _ initialChunk: [String: Any]
    ) -> EventLoopFuture<Void> {
        guard let data = try? JSONSerialization.data(withJSONObject: initialChunk) else {
            return context.eventLoop.makeFailedFuture(ServerRequestError.invalid(
                message: "stream response could not be encoded",
                param: nil,
                code: "internal_error"))
        }
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    private func enqueueToolCallChunks(id: String,
                                       created: Int,
                                       toolIndex: Int,
                                       call: ParsedToolCall,
                                       outbox: SSEOutbox,
                                       context: ChannelHandlerContext) {
        let fragments = utf8Fragments(call.argumentsJSON, maximumBytes: 1024)
        for (index, fragment) in fragments.enumerated() {
            var function: [String: Any] = ["arguments": fragment]
            var tool: [String: Any] = ["index": toolIndex, "function": function]
            if index == 0 {
                function["name"] = call.name
                tool["id"] = call.id
                tool["type"] = "function"
                tool["function"] = function
            }
            enqueueStreamChunk(
                chunk(id: id, created: created,
                      delta: ["tool_calls": [tool]],
                      finishReason: nil),
                outbox: outbox,
                context: context)
        }
    }

    private func finishStream(_ context: ChannelHandlerContext,
                              id: String,
                              created: Int,
                              completion: ServerCompletion,
                              includeUsage: Bool,
                              outbox: SSEOutbox) {
        if let frame = streamFrame(
            chunk(id: id, created: created,
                  delta: [:],
                  finishReason: completion.finishReason)) {
            _ = outbox.enqueue(frame)
        }
        if includeUsage,
           let frame = streamFrame([
               "id": id,
               "object": "chat.completion.chunk",
               "created": created,
               "model": responseModelID,
               "choices": [],
               "usage": usageObject(completion.usage),
           ]) {
            _ = outbox.enqueue(frame)
        }
        outbox.enqueueTerminal([Self.doneFrame()], closeWhenDrained: false)
    }

    private func chunk(id: String,
                       created: Int,
                       delta: [String: Any],
                       finishReason: String?) -> [String: Any] {
        let encodedReason: Any = finishReason.map { $0 as Any } ?? NSNull()
        return [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": responseModelID,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": encodedReason,
            ]],
        ]
    }

    /// Enqueue one SSE frame. Encoding or backpressure failure fails the
    /// stream with a terminal frame instead of silently dropping the chunk
    /// (S4/S5).
    private func enqueueStreamChunk(_ object: [String: Any],
                                    outbox: SSEOutbox,
                                    context: ChannelHandlerContext) {
        guard let frame = streamFrame(object) else {
            failStream(outbox: outbox, context: context,
                       message: "stream response could not be encoded",
                       code: "internal_error")
            return
        }
        guard outbox.enqueue(frame) else {
            failStream(outbox: outbox, context: context,
                       message: "stream backpressure limit exceeded; client is too slow",
                       code: "stream_overflow")
            return
        }
    }

    private func streamFrame(_ object: [String: Any]) -> Data? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return nil
        }
        return Self.sseFrame("data: " + String(decoding: data, as: UTF8.self))
    }

    private func failStream(outbox: SSEOutbox,
                            context: ChannelHandlerContext,
                            message: String,
                            code: String,
                            surface: APISurface = .chat) {
        let envelope = OpenAIErrorEnvelope(message: message,
                                           code: code,
                                           type: "server_error")
        outbox.enqueueTerminal(Self.failureFrames(envelope, surface: surface),
                               closeWhenDrained: true)
        activeTask?.cancel()
    }

    /// Drain the outbox with backpressure: each chunk's write future is
    /// awaited, so a slow reader stalls here (NIO holds the bytes in its
    /// outbound buffer) instead of letting pending memory grow without bound
    /// (S4). Write failures cancel the generation and close the connection.
    private func drainOutbox(_ context: ChannelHandlerContext,
                             outbox: SSEOutbox) async {
        while let frame = await outbox.next() {
            do {
                try await writeSSEChunk(context, frame)
            } catch {
                // The write failed (client gone or socket error): nothing more
                // can be delivered. Cancel the generation and close.
                activeTask?.cancel()
                context.close(promise: nil)
                return
            }
        }
        // Outbox closed and drained: end the HTTP response body; close the
        // connection when the stream ended in failure.
        let endWriteFailed: Bool
        do {
            try await writeSSEEnd(context)
            endWriteFailed = false
        } catch {
            endWriteFailed = true
        }
        let close = outbox.closeWhenDrained
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            if self.inFlightRequests > 0 { self.inFlightRequests -= 1 }
            self.resetIdleDeadline(contextBox.value)
            if close || endWriteFailed {
                contextBox.value.close(promise: nil)
            }
        }
    }

    private func writeSSEChunk(_ context: ChannelHandlerContext,
                               _ frame: Data) async throws {
        let promise = context.eventLoop.makePromise(of: Void.self)
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var buffer = contextBox.value.channel.allocator.buffer(capacity: frame.count)
            buffer.writeBytes(frame)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: promise)
        }
        try await promise.futureResult.get()
    }

    private func writeSSEEnd(_ context: ChannelHandlerContext) async throws {
        let promise = context.eventLoop.makePromise(of: Void.self)
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.end(nil)), promise: promise)
        }
        try await promise.futureResult.get()
    }

    private func writeHeartbeat(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: ": ping\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    private func writeHeadOnly(_ context: ChannelHandlerContext,
                               status: HTTPResponseStatus) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "0")
            headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil))).whenFailure { _ in
                contextBox.value.close(promise: nil)
            }
            if self.inFlightRequests > 0 { self.inFlightRequests -= 1 }
            self.resetIdleDeadline(contextBox.value)
        }
    }

    private func writeCodable<T: Encodable>(_ context: ChannelHandlerContext,
                                            status: HTTPResponseStatus,
                                            _ value: T,
                                            extraHeaders: [(String, String)] = []) {
        guard let data = try? JSONEncoder().encode(value) else {
            // S5: encoding failure must not silently drop the response; send a
            // minimal error envelope instead.
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        writeData(context, status: status, data: data, extraHeaders: extraHeaders)
    }

    private func writeError(_ context: ChannelHandlerContext,
                            status: HTTPResponseStatus,
                            _ error: OpenAIErrorEnvelope) {
        writeCodable(context, status: status, error)
    }

    private func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        writeData(context, status: status, data: data)
    }

    private func writeData(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           data: Data,
                           extraHeaders: [(String, String)] = []) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
            for (name, value) in extraHeaders {
                headers.add(name: name, value: value)
            }
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            // S5: a failed response write leaves no terminal frame possible;
            // close the connection so the client never hangs.
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil))).whenFailure { _ in
                contextBox.value.close(promise: nil)
            }
            if self.inFlightRequests > 0 { self.inFlightRequests -= 1 }
            self.resetIdleDeadline(contextBox.value)
        }
    }

    /// S1: (re)arm the per-connection idle close deadline. Fires after
    /// `idleReadTimeout` without read activity; the connection is closed only
    /// when no request is in flight (idle keep-alive or a stalled slowloris
    /// request), never in the middle of a generation.
    private func resetIdleDeadline(_ context: ChannelHandlerContext) {
        idleCloseTask?.cancel()
        let contextBox = SendableContext(context)
        idleCloseTask = context.eventLoop.scheduleTask(
            in: NVMAIHTTPServer.idleReadTimeout) {
            if self.inFlightRequests == 0 {
                contextBox.value.close(promise: nil)
            }
        }
    }

    private static func awaitDrainer(_ drainer: Task<Void, Never>) async {
        await withTaskCancellationHandler {
            await drainer.value
        } onCancel: {
            drainer.cancel()
        }
    }

    private static func sseFrame(_ text: String) -> Data {
        var bytes = Data(text.utf8)
        bytes.append(Data("\n\n".utf8))
        return bytes
    }

    private static func doneFrame() -> Data {
        sseFrame("data: [DONE]")
    }

    private static func errorFrame(_ envelope: OpenAIErrorEnvelope) -> Data? {
        guard let data = try? JSONEncoder().encode(envelope) else { return nil }
        return sseFrame("data: " + String(decoding: data, as: UTF8.self))
    }

    private func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "prompt_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens,
            ],
        ]
    }

    private func toolCallObject(_ call: ParsedToolCall) -> [String: Any] {
        [
            "id": call.id,
            "type": "function",
            "function": [
                "name": call.name,
                "arguments": call.argumentsJSON,
            ],
        ]
    }

    private func utf8Fragments(_ text: String, maximumBytes: Int) -> [String] {
        guard !text.isEmpty else { return [""] }
        var result: [String] = []
        var current = ""
        var bytes = 0
        for character in text {
            let size = String(character).utf8.count
            if bytes + size > maximumBytes, !current.isEmpty {
                result.append(current)
                current = ""
                bytes = 0
            }
            current.append(character)
            bytes += size
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}

/// Bounded FIFO of pre-encoded SSE frames for one stream, with a cap that
/// fails the stream when a slow reader outruns the drainer (S4).
/// unchecked-invariant: every field is guarded by `lock`. The queue is written
/// from the generation task and drained from the event loop, which is exactly
/// why the lock is here rather than relying on loop confinement.
private final class SSEOutbox: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [Data] = []
    private var pendingDrain: CheckedContinuation<Data?, Never>?
    private var closed = false
    private var overflowed = false
    private var closeAfterDrain = false
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
    }

    var closeWhenDrained: Bool {
        lock.withLock { closeAfterDrain }
    }

    /// Enqueue a regular content frame. Returns false when the outbox is
    /// closed, already failed, or the cap is exceeded (slow reader).
    func enqueue(_ frame: Data) -> Bool {
        lock.withLock {
            guard !closed, !overflowed else { return false }
            if frames.count >= capacity {
                overflowed = true
                return false
            }
            push(frame)
            return true
        }
    }

    /// Enqueue terminal frames ([DONE] / error) and close the outbox. Later
    /// frames are rejected; the drainer writes everything already queued in
    /// order and then (when `closeWhenDrained`) closes the connection.
    func enqueueTerminal(_ frames: [Data], closeWhenDrained: Bool) {
        lock.withLock {
            guard !closed else { return }
            for frame in frames { push(frame) }
            closed = true
            if closeWhenDrained { closeAfterDrain = true }
        }
    }

    /// Await the next frame; nil once the outbox is closed and drained.
    func next() async -> Data? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.withLock {
                    if !frames.isEmpty {
                        continuation.resume(returning: frames.removeFirst())
                    } else if closed {
                        continuation.resume(returning: nil)
                    } else {
                        pendingDrain = continuation
                    }
                }
            }
        } onCancel: {
            self.cancelPendingDrain()
        }
    }

    private func push(_ frame: Data) {
        if let continuation = pendingDrain {
            // Hand the frame directly to the awaiting drainer; do NOT also
            // append it to frames, or the drainer's next() would remove and
            // deliver the same frame a second time.
            pendingDrain = nil
            continuation.resume(returning: frame)
        } else {
            frames.append(frame)
        }
    }

    private func cancelPendingDrain() {
        lock.withLock {
            if let continuation = pendingDrain {
                pendingDrain = nil
                continuation.resume(returning: nil)
            }
        }
    }
}

private final class ChildChannelRegistry: Sendable {
    private struct State {
        var channels: [ObjectIdentifier: Channel] = [:]
        var tasks: [UUID: Task<Void, Never>] = [:]
        var shuttingDown = false
    }

    private let state = Mutex(State())
    private let maximumChannels: Int

    init(maximumChannels: Int) {
        self.maximumChannels = maximumChannels
    }

    func insert(_ channel: Channel) {
        let shouldClose = state.withLock {
            guard !$0.shuttingDown else { return true }
            // S1: connection cap — reject beyond maximumChannels.
            if $0.channels.count >= maximumChannels {
                return true
            }
            $0.channels[ObjectIdentifier(channel)] = channel
            return false
        }
        if shouldClose {
            channel.close(promise: nil)
        }
    }

    func remove(_ channel: Channel) {
        _ = state.withLock {
            $0.channels.removeValue(forKey: ObjectIdentifier(channel))
        }
    }

    func startTask(
        _ operation: @escaping @Sendable () async -> Void
    ) -> Task<Void, Never> {
        state.withLock { state in
            let id = UUID()
            let task = Task { [self] in
                defer {
                    _ = self.state.withLock {
                        $0.tasks.removeValue(forKey: id)
                    }
                }
                await operation()
            }
            state.tasks[id] = task
            if state.shuttingDown {
                task.cancel()
            }
            return task
        }
    }

    func closeAll() async {
        let channels = state.withLock {
            $0.shuttingDown = true
            return Array($0.channels.values)
        }
        for channel in channels {
            try? await channel.close().get()
        }
        let tasks = state.withLock { Array($0.tasks.values) }
        for task in tasks {
            task.cancel()
        }
        for task in tasks {
            await task.value
        }
    }

    var count: Int {
        state.withLock { $0.channels.count }
    }
}

/// unchecked-invariant: a transport for one `ChannelHandlerContext` across a
/// `@Sendable` boundary. The context itself is NOT thread-safe -- every use of
/// `.value` below hops back to the channel's event loop first (writeJSON,
/// writeHeartbeat, and `.eventLoop` which is itself immutable). The box makes
/// the capture legal; the event-loop hop is what makes it correct.
private final class SendableContext: @unchecked Sendable {
    let value: ChannelHandlerContext

    init(_ value: ChannelHandlerContext) {
        self.value = value
    }
}

private final class RequestPhaseState: Sendable {
    private let state = Mutex("accepted")

    var value: String { state.withLock { $0 } }

    func set(_ value: String) {
        state.withLock { $0 = value }
    }
}

/// unchecked-invariant: every field is guarded by `lock`. Start/stop are
/// driven from the generation task while the heartbeat fires on the event loop,
/// so both sides contend and neither can rely on loop confinement.
private final class StreamState: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    private var stopped = false
    private var heartbeat: RepeatedTask?
    private var startFuture: EventLoopFuture<Void>?
    private var toolIndex = 0

    var isStarted: Bool {
        lock.withLock { started }
    }

    func start(eventLoop: EventLoop,
               interval: TimeAmount,
               ping: @escaping @Sendable () -> Void) -> Bool {
        lock.withLock {
            guard !started else { return false }
            started = true
            stopped = false
            startFuture = nil
            heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: interval,
                delay: interval) { [weak self] _ in
                    guard self?.shouldPing == true else { return }
                    ping()
                }
            return true
        }
    }

    func setStartFuture(_ future: EventLoopFuture<Void>) {
        lock.withLock { startFuture = future }
    }

    func waitUntilStarted() async throws {
        let future = lock.withLock { startFuture }
        if let future {
            try await future.get()
        }
    }

    private var shouldPing: Bool {
        lock.withLock { started && !stopped }
    }

    func stop() {
        lock.withLock {
            stopped = true
            heartbeat?.cancel()
            heartbeat = nil
        }
    }

    func nextToolIndex() -> Int {
        lock.withLock {
            defer { toolIndex += 1 }
            return toolIndex
        }
    }
}
