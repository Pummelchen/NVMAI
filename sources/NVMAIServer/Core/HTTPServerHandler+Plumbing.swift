//
//  HTTPServerHandler+Plumbing.swift
//  NVMAIServer
//
//  Response writing, SSE framing, streaming and the idle/deadline plumbing shared by
//  every surface.
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import NVMAI

extension ServerHTTPHandler {
    // MARK: Shared response plumbing

    /// Start an SSE response: the head, then whatever frames the surface
    /// opens with.
    func writeStreamHead(_ context: ChannelHandlerContext,
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
    func writeRequestError(_ context: ChannelHandlerContext,
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

    func writeJSON(_ context: ChannelHandlerContext,
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
    func handleAsyncFailure(_ error: Error,
                                    context: ChannelHandlerContext,
                                    id: String,
                                    phase: String,
                                    stream: Bool,
                                    outbox: SSEOutbox?,
                                    streamState: StreamState,
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
        // The stream branch is only correct once the SSE head exists -- it is
        // written by `startStream`, which the coordinator calls when it admits
        // the request. A rejection that happens *before* admission (queue full,
        // shutting down) never gets there, and the frames queued here were then
        // written as a body with no status line: the client saw `data: {...}`
        // bytes where a 429 should have been, on every streaming surface.
        // Falling through writes a real response instead.
        if stream, let outbox, !streamState.isStarted {
            // Refused before admission, so no SSE head was ever written and the
            // paths below write this response in full. Retire the outbox so its
            // drainer neither waits forever nor appends an `end` to a response it
            // did not produce.
            outbox.abandon()
        }
        if stream, let outbox, streamState.isStarted {
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
    func enqueueChatEvent(_ event: ServerInferenceEvent,
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

    func writeCompletion(_ context: ChannelHandlerContext,
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

    func beginStream(
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

    func enqueueToolCallChunks(id: String,
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

    func finishStream(_ context: ChannelHandlerContext,
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

    func chunk(id: String,
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
    func enqueueStreamChunk(_ object: [String: Any],
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

    func streamFrame(_ object: [String: Any]) -> Data? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return nil
        }
        return Self.sseFrame("data: " + String(decoding: data, as: UTF8.self))
    }

    func failStream(outbox: SSEOutbox,
                            context: ChannelHandlerContext,
                            message: String,
                            code: String,
                            surface: APISurface = .chat) {
        let envelope = OpenAIErrorEnvelope(message: message,
                                           code: code,
                                           type: "server_error")
        outbox.enqueueTerminal(Self.failureFrames(envelope, surface: surface),
                               closeWhenDrained: true)
        // Cancel the generation this runs in, by identity: `failStream` is called
        // from the generation's own task (the event callback), so this is exact,
        // and it is not `activeTask` -- a pipelined follow-up request may already
        // have replaced that. The frames above are delivered by the drainer,
        // which runs in a separate task and is unaffected.
        withUnsafeCurrentTask { $0?.cancel() }
    }

    /// Drain the outbox with backpressure: each chunk's write future is
    /// awaited, so a slow reader stalls here (NIO holds the bytes in its
    /// outbound buffer) instead of letting pending memory grow without bound
    /// (S4). Write failures cancel the generation and close the connection.
    func drainOutbox(_ context: ChannelHandlerContext,
                             outbox: SSEOutbox,
                             streamState: StreamState) async {
        while let frame = await outbox.next() {
            do {
                try await writeSSEChunk(context, frame)
            } catch {
                // The write failed (client gone or socket error): nothing more
                // can be delivered. Cancel the generation and close -- but only
                // when the slot still holds the generation this drainer belongs
                // to. This drainer runs in its own task, so `activeTask` is the
                // only handle it has, and under pipelining that is the *next*
                // request's task; cancelling it would stop the wrong generation.
                if inFlightRequests <= 1 { activeTask?.cancel() }
                context.close(promise: nil)
                return
            }
        }
        // Outbox closed and drained. End the HTTP response body — unless this
        // outbox was abandoned, which means the request was refused before a
        // stream head existed and the error response has already been written in
        // full: an `end` here would be a second one on the same request.
        let endWriteFailed: Bool
        if outbox.isAbandoned {
            endWriteFailed = false
        } else {
            // Stop the heartbeat *before* the terminal `end`.
            //
            // `writeHeartbeat` writes a body part gated only on
            // `started && !stopped`, and `stop()` otherwise runs in the request
            // task's `defer` — which is *after* this write. A ping landing in
            // between is a body with no head outstanding: NIO's
            // `HTTPServerProtocolErrorHandler` traps on that, and in release it
            // is a malformed response. The test suite schedules heartbeats every
            // 10 ms, so the window was being hit intermittently; production's 5 s
            // interval only makes it rarer.
            streamState.stop()
            do {
                try await writeSSEEnd(context)
                endWriteFailed = false
            } catch {
                endWriteFailed = true
            }
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

    func writeSSEChunk(_ context: ChannelHandlerContext,
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

    func writeSSEEnd(_ context: ChannelHandlerContext) async throws {
        let promise = context.eventLoop.makePromise(of: Void.self)
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.end(nil)), promise: promise)
        }
        try await promise.futureResult.get()
    }

    func writeHeartbeat(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: ": ping\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    func writeHeadOnly(_ context: ChannelHandlerContext,
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

    func writeCodable<T: Encodable>(_ context: ChannelHandlerContext,
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

    func writeError(_ context: ChannelHandlerContext,
                            status: HTTPResponseStatus,
                            _ error: OpenAIErrorEnvelope) {
        writeCodable(context, status: status, error)
    }

    func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        writeData(context, status: status, data: data)
    }

    func writeData(_ context: ChannelHandlerContext,
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
    func resetIdleDeadline(_ context: ChannelHandlerContext) {
        idleCloseTask?.cancel()
        let contextBox = SendableContext(context)
        idleCloseTask = context.eventLoop.scheduleTask(
            in: NVMAIHTTPServer.idleReadTimeout) {
            if self.inFlightRequests == 0 {
                contextBox.value.close(promise: nil)
            }
        }
    }

    static func awaitDrainer(_ drainer: Task<Void, Never>) async {
        await withTaskCancellationHandler {
            await drainer.value
        } onCancel: {
            drainer.cancel()
        }
    }

    static func sseFrame(_ text: String) -> Data {
        var bytes = Data(text.utf8)
        bytes.append(Data("\n\n".utf8))
        return bytes
    }

    static func doneFrame() -> Data {
        sseFrame("data: [DONE]")
    }

    static func errorFrame(_ envelope: OpenAIErrorEnvelope) -> Data? {
        guard let data = try? JSONEncoder().encode(envelope) else { return nil }
        return sseFrame("data: " + String(decoding: data, as: UTF8.self))
    }

    func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "prompt_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens,
            ],
        ]
    }

    func toolCallObject(_ call: ParsedToolCall) -> [String: Any] {
        [
            "id": call.id,
            "type": "function",
            "function": [
                "name": call.name,
                "arguments": call.argumentsJSON,
            ],
        ]
    }

    func utf8Fragments(_ text: String, maximumBytes: Int) -> [String] {
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
