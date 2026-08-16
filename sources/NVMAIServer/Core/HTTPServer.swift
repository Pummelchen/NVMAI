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
    private let childChannels = ChildChannelRegistry(
        maximumChannels: maximumConcurrentConnections)
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    public init(modelID: String,
                queueLimit: Int,
                backend: any ServerInferenceBackend,
                heartbeatInterval: TimeAmount = .seconds(5),
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1)) {
        self.group = group
        self.modelID = modelID
        self.backend = backend
        self.coordinator = ServerCoordinator(queueLimit: queueLimit)
        self.heartbeatInterval = heartbeatInterval
    }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let backend = self.backend
        let coordinator = self.coordinator
        let heartbeatInterval = self.heartbeatInterval
        let childChannels = self.childChannels
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
                        childChannels: childChannels))
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
    private let childChannels: ChildChannelRegistry
    private var head: HTTPRequestHead?
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
         childChannels: ChildChannelRegistry) {
        self.modelID = modelID
        self.backend = backend
        self.coordinator = coordinator
        self.heartbeatInterval = heartbeatInterval
        self.childChannels = childChannels
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
        switch (head.method, path) {
        case (.GET, "/health"):
            writeJSON(context, status: .ok, object: ["status": "ok"])
        case (.GET, "/v1/models"):
            // Janeway: "Is it? You've scanned our vessel. You know we can
            // match your firepower." — here is the model inventory.
            // Advertise the base model plus the "<model>-fast" alias, which
            // serves the same weights with the CLI-strip heuristic enabled.
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
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleCompletion(
                body: body,
                context: context)
        case (.POST, "/v1/responses"):
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleResponses(
                body: body,
                context: context)
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"), (_, "/v1/responses"):
            writeError(context, status: .methodNotAllowed,
                       OpenAIErrorEnvelope(message: "method not allowed",
                                           code: "method_not_allowed"))
        default:
            writeError(context, status: .notFound,
                       OpenAIErrorEnvelope(message: "route not found",
                                           code: "not_found"))
        }
    }

    private func handleCompletion(body: ByteBuffer,
                                  context: ChannelHandlerContext) {
        do {
            // Picard: "I am Locutus of Borg. Resistance is futile. Your life
            // as it has been is over." The raw request is assimilated here.
            // One copy out of the ByteBuffer, not two: a [UInt8] hop would
            // duplicate a body of up to `maximumBodyBytes` before decoding.
            let decoded = try JSONDecoder().decode(
                OpenAIChatRequest.self, from: Data(body.readableBytesView))
            let request = try OpenAIRequestValidator.validate(
                decoded, modelID: modelID, maxContext: backend.maximumContext)
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
                            switch event {
                            case .content(let text):
                                self.enqueueStreamChunk(
                                    self.chunk(id: responseID, created: created,
                                               delta: ["content": text],
                                               finishReason: nil),
                                    outbox: outbox,
                                    context: contextBox.value)
                            case .toolCall(let call):
                                self.enqueueToolCallChunks(
                                    id: responseID,
                                    created: created,
                                    toolIndex: streamState.nextToolIndex(),
                                    call: call,
                                    outbox: outbox,
                                    context: contextBox.value)
                            }
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
                    self.handleAsyncError(error,
                                          context: contextBox.value,
                                          id: responseID,
                                          phase: phaseState.value,
                                          stream: request.stream,
                                          outbox: outbox)
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

    /// Streaming item bookkeeping for one /v1/responses turn (message item
    /// announced, function-call output indices).
    private final class ResponsesItemState: @unchecked Sendable {
        var messageAnnounced = false
        var toolCount = 0
    }

    private static func eventFrame(name: String, object: [String: Any]) -> Data? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return nil
        }
        return Self.sseFrame("event: " + name + "\ndata: " + String(decoding: data, as: UTF8.self))
    }

    /// OpenAI Responses API endpoint (`POST /v1/responses`). The request is
    /// mapped onto the chat-completions path (see ResponsesAPIMapper) and the
    /// generation is streamed back as Responses-API SSE events (or returned
    /// as a single response object when stream is false). This is what lets
    /// current Codex CLI versions (which only speak the Responses API) talk
    /// to NVMAI directly.
    private func handleResponses(body: ByteBuffer,
                                 context: ChannelHandlerContext) {
        do {
            let decoded = try JSONDecoder().decode(
                ResponsesAPIRequest.self, from: Data(body.readableBytesView))
            if decoded.store == true {
                throw ServerRequestError.invalid(
                    message: "store is not supported",
                    param: "store", code: "unsupported_value")
            }
            let chatRequest = try ResponsesAPIMapper.chatRequest(decoded)
            let request = try OpenAIRequestValidator.validate(
                chatRequest, modelID: modelID, maxContext: backend.maximumContext)
            let responseID = ResponsesAPIBuilder.responseID()
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let itemState = ResponsesItemState()
            let phaseState = requestPhaseState
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeHeartbeat(contextBox.value)
                      }) else { return }
                let future = self.beginResponsesStream(
                    contextBox.value,
                    id: responseID,
                    created: created,
                    request: request,
                    store: decoded.store)
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
                            switch event {
                            case .content(let text):
                                self.enqueueResponsesContentDelta(
                                    id: responseID, created: created,
                                    request: request, text: text,
                                    itemState: itemState,
                                    outbox: outbox, context: contextBox.value)
                            case .toolCall(let call):
                                self.enqueueResponsesToolDelta(
                                    id: responseID, created: created,
                                    request: request, call: call,
                                    itemState: itemState,
                                    outbox: outbox, context: contextBox.value)
                            }
                        }
                    }
                    ServerLog.completed(id: responseID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    if request.stream, let outbox {
                        streamState.stop()
                        self.finishResponsesStream(
                            contextBox.value, id: responseID, created: created,
                            request: request, completion: completion,
                            itemState: itemState, outbox: outbox)
                    } else {
                        self.writeResponses(
                            contextBox.value, id: responseID, created: created,
                            request: request, completion: completion)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncError(error,
                                          context: contextBox.value,
                                          id: responseID,
                                          phase: phaseState.value,
                                          stream: request.stream,
                                          outbox: outbox)
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

    private func beginResponsesStream(_ context: ChannelHandlerContext,
                                      id: String,
                                      created: Int,
                                      request: ValidatedChatRequest,
                                      store: Bool?) -> EventLoopFuture<Void> {
        let response = ResponsesAPIBuilder.responseObject(
            id: id, created: created, model: modelID, status: "in_progress",
            output: [], usage: nil, store: store ?? false)
        let createdEvent = ResponsesAPIBuilder.responseObject(
            id: id, created: created, model: modelID, status: "in_progress",
            output: [], usage: nil, store: store ?? false)
        var frames = Data()
        if let frame = Self.eventFrame(name: "response.created",
                                       object: ["type": "response.created", "response": response]),
           let second = Self.eventFrame(name: "response.in_progress",
                                        object: ["type": "response.in_progress", "response": createdEvent]) {
            frames.append(frame)
            frames.append(second)
        }
        let framesSnapshot = frames
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)), promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: framesSnapshot.count)
            buffer.writeBytes(framesSnapshot)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    private func enqueueResponsesEvent(name: String,
                                       object: [String: Any],
                                       outbox: SSEOutbox,
                                       context: ChannelHandlerContext) {
        guard let frame = Self.eventFrame(name: name, object: object) else {
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

    private func enqueueResponsesContentDelta(id: String,
                                              created: Int,
                                              request: ValidatedChatRequest,
                                              text: String,
                                              itemState: ResponsesItemState,
                                              outbox: SSEOutbox,
                                              context: ChannelHandlerContext) {
        if !itemState.messageAnnounced {
            itemState.messageAnnounced = true
            enqueueResponsesEvent(
                name: "response.output_item.added",
                object: ["type": "response.output_item.added", "output_index": 0,
                         "item": ["id": id + "_msg0", "type": "message",
                                  "role": "assistant", "status": "in_progress",
                                  "content": []]],
                outbox: outbox, context: context)
            enqueueResponsesEvent(
                name: "response.content_part.added",
                object: ["type": "response.content_part.added", "item_id": id + "_msg0",
                         "output_index": 0, "content_index": 0,
                         "part": ["type": "output_text", "text": "", "annotations": []]],
                outbox: outbox, context: context)
        }
        enqueueResponsesEvent(
            name: "response.output_text.delta",
            object: ["type": "response.output_text.delta", "item_id": id + "_msg0",
                     "output_index": 0, "content_index": 0, "delta": text],
            outbox: outbox, context: context)
        enqueueResponsesEvent(
            name: "response.content_part.delta",
            object: ["type": "response.content_part.delta", "item_id": id + "_msg0",
                     "output_index": 0, "content_index": 0,
                     "delta": ["type": "output_text", "text": text, "annotations": []]],
            outbox: outbox, context: context)
    }

    private func enqueueResponsesToolDelta(id: String,
                                           created: Int,
                                           request: ValidatedChatRequest,
                                           call: ParsedToolCall,
                                           itemState: ResponsesItemState,
                                           outbox: SSEOutbox,
                                           context: ChannelHandlerContext) {
        let index = itemState.toolCount
        itemState.toolCount += 1
        let outputIndex = index + 1
        let itemID = id + "_fc\(index)"
        enqueueResponsesEvent(
            name: "response.output_item.added",
            object: ["type": "response.output_item.added", "output_index": outputIndex,
                     "item": ["id": itemID, "type": "function_call",
                              "status": "in_progress", "name": call.name,
                              "arguments": "", "call_id": call.id,
                              "output_index": outputIndex]],
            outbox: outbox, context: context)
        let fragments = utf8Fragments(call.argumentsJSON, maximumBytes: 1024)
        for fragment in fragments {
            enqueueResponsesEvent(
                name: "response.function_call_arguments.delta",
                object: ["type": "response.function_call_arguments.delta",
                         "item_id": itemID, "output_index": outputIndex, "delta": fragment],
                outbox: outbox, context: context)
        }
    }

    private func finishResponsesStream(_ context: ChannelHandlerContext,
                                       id: String,
                                       created: Int,
                                       request: ValidatedChatRequest,
                                       completion: ServerCompletion,
                                       itemState: ResponsesItemState,
                                       outbox: SSEOutbox) {
        if itemState.messageAnnounced {
            enqueueResponsesEvent(
                name: "response.content_part.done",
                object: ["type": "response.content_part.done", "item_id": id + "_msg0",
                         "output_index": 0, "content_index": 0,
                         "part": ["type": "output_text", "text": completion.content,
                                  "annotations": []]],
                outbox: outbox, context: context)
            enqueueResponsesEvent(
                name: "response.output_item.done",
                object: ["type": "response.output_item.done", "output_index": 0,
                         "item": ResponsesAPIBuilder.messageItem(
                            id: id + "_msg0", role: "assistant",
                            text: completion.content, status: "completed")],
                outbox: outbox, context: context)
        }
        for (index, call) in completion.toolCalls.enumerated() {
            let outputIndex = index + 1
            let itemID = id + "_fc\(index)"
            enqueueResponsesEvent(
                name: "response.function_call_arguments.done",
                object: ["type": "response.function_call_arguments.done",
                         "item_id": itemID, "output_index": outputIndex,
                         "arguments": call.argumentsJSON],
                outbox: outbox, context: context)
            enqueueResponsesEvent(
                name: "response.output_item.done",
                object: ["type": "response.output_item.done", "output_index": outputIndex,
                         "item": ResponsesAPIBuilder.functionCallItem(
                            id: itemID, name: call.name,
                            arguments: call.argumentsJSON, callID: call.id,
                            outputIndex: outputIndex, status: "completed")],
                outbox: outbox, context: context)
        }
        var output = ResponsesAPIBuilder.outputItems(completion: completion, idPrefix: id)
        if output.isEmpty {
            output.append(ResponsesAPIBuilder.messageItem(
                id: id + "_msg0", role: "assistant", text: "", status: "completed"))
        }
        if let frame = Self.eventFrame(
            name: "response.completed",
            object: ["type": "response.completed",
                     "response": ResponsesAPIBuilder.responseObject(
                        id: id, created: created, model: modelID, status: "completed",
                        output: output, usage: completion.usage)]) {
            _ = outbox.enqueue(frame)
        }
        outbox.enqueueTerminal([Self.doneFrame()], closeWhenDrained: false)
    }

    private func writeResponses(_ context: ChannelHandlerContext,
                                id: String,
                                created: Int,
                                request: ValidatedChatRequest,
                                completion: ServerCompletion) {
        let output = ResponsesAPIBuilder.outputItems(completion: completion, idPrefix: id)
        writeJSON(context, status: .ok, object:
                  ResponsesAPIBuilder.responseObject(
                    id: id, created: created, model: modelID, status: "completed",
                    output: output, usage: completion.usage))
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
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map(toolCallObject)
        }
        let object: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": modelID,
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
               "model": modelID,
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
            "model": modelID,
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
                            code: String) {
        let envelope = OpenAIErrorEnvelope(message: message,
                                           code: code,
                                           type: "server_error")
        outbox.enqueueTerminal(
            Self.errorFrame(envelope).map { [$0, Self.doneFrame()] } ?? [Self.doneFrame()],
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

    private func handleAsyncError(_ error: Error,
                                  context: ChannelHandlerContext,
                                  id: String,
                                  phase: String,
                                  stream: Bool,
                                  outbox: SSEOutbox?) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            switch requestError {
            case .queueFull: status = .tooManyRequests
            case .unknownModel: status = .notFound
            default: status = .badRequest
            }
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
            // Cancellation (client disconnect / shutdown) ends the stream
            // cleanly with [DONE]; real failures emit an error event first and
            // the connection is closed after the frames drain.
            if error is CancellationError {
                outbox.enqueueTerminal([Self.doneFrame()], closeWhenDrained: true)
            } else {
                outbox.enqueueTerminal(
                    Self.errorFrame(envelope).map { [$0, Self.doneFrame()] } ?? [Self.doneFrame()],
                    closeWhenDrained: true)
            }
            return
        }
        if error is CancellationError {
            // S20: the client disconnected or the server is shutting down;
            // there is no one to write to. Do not emit a misleading 500.
            return
        }
        writeError(context, status: status, envelope)
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
                                            _ value: T) {
        guard let data = try? JSONEncoder().encode(value) else {
            // S5: encoding failure must not silently drop the response; send a
            // minimal error envelope instead.
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        writeData(context, status: status, data: data)
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
                           data: Data) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
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
