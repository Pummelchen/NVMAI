import Foundation
import Testing
@testable import NVMAI
@testable import NVMAIServerCore

private func send(_ port: Int, _ method: String, _ path: String, json: String? = nil,
                  headers: [String: String] = [:]) async throws -> (Data, HTTPURLResponse) {
    var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)\(path)")))
    request.httpMethod = method
    if let json {
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(json.utf8)
    }
    for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
    let (data, response) = try await URLSession.shared.data(for: request)
    return (data, try #require(response as? HTTPURLResponse))
}

private func object(_ data: Data) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

private func chat(_ model: String, extra: String = "") -> String {
    #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]\#(extra)}"#
}

private func withServer<T>(backend: any ServerInferenceBackend,
                           router: (any ModelRouting)? = nil,
                           queueLimit: Int = 4,
                           _ body: (Int) async throws -> T) async throws -> T {
    let server = NVMAIHTTPServer(modelID: "test-model", queueLimit: queueLimit,
                                 backend: backend, router: router)
    let channel = try await server.start(port: 0)
    let port = try #require(channel.localAddress?.port)
    do {
        let result = try await body(port)
        try await server.shutdown()
        return result
    } catch {
        try await server.shutdown()
        throw error
    }
}

private func withRouter<T>(log: RoutingEventLog, delay: Duration? = nil,
                           _ body: (Int, ModelRouter) async throws -> T) async throws -> T {
    let router = try RoutingFixture.router(log: log, delay: delay)
    try await router.preload()
    return try await withServer(backend: router, router: router) { port in
        try await body(port, router)
    }
}

@Suite("Dynamic serving over HTTP", .serialized)
struct DynamicServingHTTPTests {
    private let anthropic = ["anthropic-version": "2023-06-01"]

    @Test func modelsListsEveryCatalogModelInTheOpenAIShape() async throws {
        try await withRouter(log: RoutingEventLog()) { port, _ in
            let (data, response) = try await send(port, "GET", "/v1/models")
            #expect(response.statusCode == 200)
            let list = try object(data)
            #expect(list["object"] as? String == "list")
            let models = try #require(list["data"] as? [[String: Any]])
            #expect(models.compactMap { $0["id"] as? String }
                    == ["alpha_4-Bit", "flash_8-Bit", "small-2b"])
            #expect(models.allSatisfy { $0["object"] as? String == "model" })
            #expect(models.allSatisfy { $0["owned_by"] as? String == "nvmai" })
        }
    }

    @Test func modelsAnswersInTheAnthropicShapeWhenAskedTo() async throws {
        try await withRouter(log: RoutingEventLog()) { port, _ in
            let (data, _) = try await send(port, "GET", "/v1/models", headers: anthropic)
            let list = try object(data)
            #expect(list["has_more"] as? Bool == false)
            #expect(list["first_id"] as? String == "alpha_4-Bit")
            #expect(list["last_id"] as? String == "small-2b")
            let models = try #require(list["data"] as? [[String: Any]])
            #expect(models.compactMap { $0["display_name"] as? String }
                    == ["Alpha 35B", "Flash 125B", "Small 2B"])
            #expect(models.allSatisfy { $0["type"] as? String == "model" })
            #expect(models.allSatisfy { $0["created_at"] is String })
            #expect(Set(models.flatMap(\.keys)) == ["type", "id", "display_name", "created_at"])

            let (one, status) = try await send(port, "GET", "/v1/models/small-2b", headers: anthropic)
            #expect(status.statusCode == 200)
            #expect(try object(one)["display_name"] as? String == "Small 2B")
            let (_, missing) = try await send(port, "GET", "/v1/models/nothing")
            #expect(missing.statusCode == 404)
        }
    }

    @Test func requestsRunOnTheModelTheyNameAndSayWhichItWas() async throws {
        let log = RoutingEventLog()
        try await withRouter(log: log) { port, router in
            let (data, response) = try await send(port, "POST", "/v1/chat/completions",
                                                  json: chat("small-2b"))
            #expect(response.statusCode == 200)
            let completion = try object(data)
            #expect(completion["model"] as? String == "small-2b")
            #expect(await router.residentModelID == "small-2b")

            // The "-fast" alias works for every catalog model and still
            // routes to (and reports) the base model.
            let (fast, fastResponse) = try await send(port, "POST", "/v1/chat/completions",
                                                      json: chat("flash_8-Bit-fast"))
            #expect(fastResponse.statusCode == 200)
            #expect(try object(fast)["model"] as? String == "flash_8-Bit")
            let last = try #require(log.requests.last)
            #expect(last.stripCLIPrompt)
            #expect(last.model == "flash_8-Bit")

            let (message, messageResponse) = try await send(
                port, "POST", "/v1/messages",
                json: #"{"model":"alpha_4-Bit","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
                headers: anthropic)
            #expect(messageResponse.statusCode == 200)
            #expect(try object(message)["model"] as? String == "alpha_4-Bit")
            #expect(log.loads == ["load alpha_4-Bit", "load small-2b", "load flash_8-Bit", "load alpha_4-Bit"])
        }
    }

    @Test func anUnknownModelIsRefusedOnEverySurface() async throws {
        let log = RoutingEventLog()
        try await withRouter(log: log) { port, _ in
            let (chatData, chatResponse) = try await send(port, "POST", "/v1/chat/completions",
                                                          json: chat("nothing"))
            #expect(chatResponse.statusCode == 404)
            let error = try #require(try object(chatData)["error"] as? [String: Any])
            #expect(error["code"] as? String == "model_not_found")

            let (_, responses) = try await send(port, "POST", "/v1/responses",
                                                json: #"{"model":"nothing","input":"hi"}"#)
            #expect(responses.statusCode == 404)
            let (_, messages) = try await send(
                port, "POST", "/v1/messages",
                json: #"{"model":"nothing","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
                headers: anthropic)
            #expect(messages.statusCode == 404)
            #expect(log.loads == ["load alpha_4-Bit"])
        }
    }

    /// The request's omitted values and its token bound come from the model it
    /// names, decided before that model is loaded -- not from the resident one.
    @Test func omittedSamplingAndTheTokenBoundComeFromTheNamedModel() async throws {
        let log = RoutingEventLog()
        try await withRouter(log: log) { port, router in
            // Alpha is resident; the CPU model's context is 32768.
            let (_, tooLong) = try await send(port, "POST", "/v1/chat/completions",
                                              json: chat("small-2b", extra: #","max_tokens":40000"#))
            #expect(tooLong.statusCode == 400)
            #expect(await router.residentModelID == "alpha_4-Bit")
            let (_, fits) = try await send(port, "POST", "/v1/chat/completions",
                                           json: chat("alpha_4-Bit", extra: #","max_tokens":40000"#))
            #expect(fits.statusCode == 200)

            _ = try await send(port, "POST", "/v1/chat/completions", json: chat("flash_8-Bit"))
            let flash = try #require(log.requests.last)
            #expect(flash.generationConfig.temperature == 1.0)
            #expect(flash.maximumCompletionTokens == RoutingFixture.configuredContext)

            _ = try await send(port, "POST", "/v1/chat/completions", json: chat("small-2b"))
            let small = try #require(log.requests.last)
            #expect(small.generationConfig.temperature == 0.6)
            #expect(small.maximumCompletionTokens == CPUModelBackend.contextCeiling)
        }
    }

    /// Four clients at once against the default queue limit: every one is
    /// admitted and answered, one generation at a time.
    @Test func fourConcurrentRequestsAllComplete() async throws {
        let log = RoutingEventLog()
        let backend = RoutedStubModel(id: "test-model", log: log, gate: nil,
                                      delay: .milliseconds(150))
        let defaults = try ServerArguments.parse(["--model", "/m"], environment: [:])
        try await withServer(backend: backend, queueLimit: defaults.queueLimit) { port in
            let statuses = try await withThrowingTaskGroup(of: Int.self) { group in
                for _ in 0..<4 {
                    group.addTask {
                        try await send(port, "POST", "/v1/chat/completions",
                                       json: chat("test-model")).1.statusCode
                    }
                }
                return try await group.reduce(into: []) { $0.append($1) }
            }
            #expect(statuses == [200, 200, 200, 200])
        }
        #expect(log.requests.count == 4)
        #expect(log.maxConcurrentGenerations == 1)
    }

    @Test func fourConcurrentRequestsAcrossModelsAllComplete() async throws {
        let log = RoutingEventLog()
        let names = ["alpha_4-Bit", "small-2b", "alpha_4-Bit", "flash_8-Bit"]
        try await withRouter(log: log, delay: .milliseconds(50)) { port, _ in
            let answered = try await withThrowingTaskGroup(of: (String, Int, String?).self) { group in
                for name in names {
                    group.addTask {
                        let (data, response) = try await send(port, "POST", "/v1/chat/completions",
                                                              json: chat(name))
                        return (name, response.statusCode, try object(data)["model"] as? String)
                    }
                }
                return try await group.reduce(into: []) { $0.append($1) }
            }
            #expect(answered.count == 4)
            for (name, status, model) in answered {
                #expect(status == 200)
                #expect(model == name)
            }
        }
        #expect(log.maxConcurrentGenerations == 1)
    }

    /// Without a router the server is the one it always was: one model plus
    /// its "-fast" alias, the backend's own defaults, every other name refused.
    @Test func aSingleModelServerIsUnchanged() async throws {
        let log = RoutingEventLog()
        let backend = RoutedStubModel(id: "test-model", log: log, gate: nil, delay: nil)
        try await withServer(backend: backend) { port in
            let (data, _) = try await send(port, "GET", "/v1/models")
            let ids = try #require(try object(data)["data"] as? [[String: Any]])
                .compactMap { $0["id"] as? String }
            #expect(ids == ["test-model", "test-model-fast"])

            let (_, other) = try await send(port, "POST", "/v1/chat/completions", json: chat("small-2b"))
            #expect(other.statusCode == 404)
            let (reply, ok) = try await send(port, "POST", "/v1/chat/completions", json: chat("test-model"))
            #expect(ok.statusCode == 200)
            #expect(try object(reply)["model"] as? String == "test-model")
        }
        let request = try #require(log.requests.first)
        #expect(request.generationConfig.temperature == GenerationDefaults.house.temperature)
        #expect(request.maximumCompletionTokens == backend.maximumContext)
    }
}

@Suite("Dynamic serving arguments")
struct DynamicServingArgumentTests {
    private func parse(_ input: [String]) throws -> ServerArguments {
        try ServerArguments.parse(input, environment: [:])
    }

    @Test func withoutAModelsDirectoryNothingChanges() throws {
        let arguments = try parse(["--model", "/m"])
        #expect(arguments.modelsDirectory == nil)
        #expect(!arguments.catalogOnly)
        #expect(arguments.reasoningLevel == nil)
        #expect(arguments.requestedReasoningLevel == .off)
    }

    @Test func theCatalogNeedsADirectoryButNoModel() throws {
        let arguments = try parse(["--catalog", "--models-dir", "/models"])
        #expect(arguments.catalogOnly)
        #expect(arguments.modelsDirectory == "/models")
        #expect(throws: ServerArgumentError.self) { try parse(["--catalog"]) }
        #expect(throws: ServerArgumentError.invalid("--model is required")) {
            try parse(["--models-dir", "/models"])
        }
    }

    @Test func reasoningReplacesTheOlderFlags() throws {
        #expect(try parse(["--model", "/m", "--reasoning", "high"]).requestedReasoningLevel == .high)
        #expect(try parse(["--model", "/m", "--thinking", "on"]).requestedReasoningLevel == .on)
        #expect(try parse(["--model", "/m", "--thinking", "on", "--reasoning-effort", "xhigh"])
                .requestedReasoningLevel == .xhigh)
        #expect(throws: ServerArgumentError.self) {
            try parse(["--model", "/m", "--reasoning", "high", "--thinking", "on"])
        }
        #expect(throws: ServerArgumentError.self) { try parse(["--model", "/m", "--reasoning", "loud"]) }
    }

    @Test func singleModelFlagsAreRefusedWithACatalog() {
        for flags in [["--model-id", "x"], ["--cpu"], ["--mtp-model", "/d"], ["--idle-unload-seconds", "60"]] {
            #expect(throws: ServerArgumentError.self) {
                try parse(["--models-dir", "/models", "--model", "a"] + flags)
            }
        }
    }

    /// One active plus `queueLimit` queued: the default must admit four
    /// concurrent clients without shedding any.
    @Test func theDefaultQueueAdmitsFourClients() throws {
        #expect(try parse(["--model", "/m"]).queueLimit + 1 >= 4)
    }
}
