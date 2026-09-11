//
//  HTTPServerHandler+Responses.swift
//  NVMAIServer
//
//  The OpenAI Responses API surface.
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import NVMAI

extension ServerHTTPHandler {
    func resolveReferences(_ items: [ResponsesAPIRequest.Item]) -> [ResponsesAPIRequest.Item] {
        items.map { item in
            guard item.resolvedType == "item_reference", let id = item.id,
                  let stored = responseStore.item(withID: id) else { return item }
            return stored
        }
    }

    /// The API reports a failed generation as a response object in state
    /// "failed" carrying the error, then ends the stream.
    func responsesFailureFrames(
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
    func priorConversation(
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
    func handleResponses(body: ByteBuffer,
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
                        await self.drainOutbox(contextBox.value, outbox: outbox,
                                              streamState: streamState)
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
                        streamState: streamState,
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
    func finalResponsesObject(id: String,
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

    func storeResponse(id: String,
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

    func beginResponsesStream(_ context: ChannelHandlerContext,
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
    func responsesEvent(_ type: String,
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

    func enqueueResponsesEvent(_ event: ServerInferenceEvent,
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
    func enqueueResponsesReasoningDelta(id: String,
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
    func closeResponsesReasoning(id: String,
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

    func enqueueResponsesContentDelta(id: String,
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
    func enqueueResponsesToolCall(id: String,
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
    func finishResponsesStream(_ context: ChannelHandlerContext,
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

}
