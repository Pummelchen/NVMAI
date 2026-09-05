import Foundation
import Testing
@testable import NVMAIMemory

/// The Valkey backend against a server that speaks the real protocol.
///
/// Serialized: each test binds a listening socket, and the suite is about the
/// wire, so running them concurrently would only add flakiness.
@Suite(.serialized) struct ValkeyMemoryStoreTests {
    private func makeStore(_ server: FakeValkeyServer,
                           limits: MemoryLimits = .init()) -> ValkeyMemoryStore {
        ValkeyMemoryStore(connection: ValkeyConnection(configuration: server.configuration),
                          prefix: "nvmai:mem",
                          limits: limits)
    }

    private func scope(_ workspace: String = "repo-a") throws -> MemoryScope {
        try MemoryScope(namespace: "nvmai", user: "local", workspace: workspace)
    }

    private func withServer(_ body: (FakeValkeyServer) async throws -> Void) async throws {
        let server = FakeValkeyServer()
        try server.start()
        defer { server.stop() }
        try await body(server)
    }

    @Test func connectsAndRoundTripsARecord() async throws {
        try await withServer { server in
            let store = makeStore(server)
            let scope = try scope()
            let key = try MemoryKey(validating: "decisions/sync")
            try await store.set(MemoryRecord(key: key, value: "Keep FooManager.",
                                             importance: 0.9, tags: ["sync"]), in: scope)

            let loaded = try await store.get(key, in: scope)
            #expect(loaded?.value == "Keep FooManager.")
            #expect(loaded?.importance == 0.9)
            #expect(loaded?.tags == ["sync"])
        }
    }

    @Test func recordsAndIndexUseTheScopedKeySpace() async throws {
        try await withServer { server in
            let store = makeStore(server)
            try await store.set(MemoryRecord(key: try MemoryKey(validating: "facts/a"), value: "1"),
                                in: try scope("repo-a"))

            let keys = server.storedKeys()
            // The scope tag is what isolates repositories; assert its shape
            // rather than trusting the store's own read path to prove it.
            #expect(keys.contains("nvmai:mem:{nvmai/local/repo-a}:r:facts/a"))
            let indexed = server.commandLog.contains { $0.first == "ZADD"
                && $0.dropFirst().first == "nvmai:mem:{nvmai/local/repo-a}:idx" }
            #expect(indexed)
        }
    }

    @Test func repositoriesAreIsolatedOnTheWire() async throws {
        try await withServer { server in
            let store = makeStore(server)
            let key = try MemoryKey(validating: "decisions/db")
            try await store.set(MemoryRecord(key: key, value: "postgres"), in: try scope("repo-a"))
            try await store.set(MemoryRecord(key: key, value: "sqlite"), in: try scope("repo-b"))

            #expect(try await store.get(key, in: try scope("repo-a"))?.value == "postgres")
            #expect(try await store.get(key, in: try scope("repo-b"))?.value == "sqlite")
            #expect(try await store.list(prefix: "", limit: 50, in: try scope("repo-a")).count == 1)
        }
    }

    @Test func neverIssuesKeysOrScan() async throws {
        try await withServer { server in
            let store = makeStore(server)
            let scope = try scope()
            for index in 0..<20 {
                try await store.set(MemoryRecord(key: try MemoryKey(validating: "facts/f\(index)"),
                                                 value: "v"), in: scope)
            }
            _ = try await store.search(MemoryQuery(text: "v"), in: scope)
            _ = try await store.list(prefix: "facts/", limit: 10, in: scope)
            _ = try await store.sessionInit(MemorySession(id: "s"), in: scope)

            // A database-wide scan is the failure this design exists to
            // prevent: one scope's cost must not depend on the others.
            let dangerous = server.commandLog.filter {
                ["KEYS", "SCAN", "FLUSHDB", "FLUSHALL"].contains($0.first?.uppercased() ?? "")
            }
            #expect(dangerous.isEmpty)
        }
    }

    @Test func listAndSearchComeBackThroughTheIndex() async throws {
        try await withServer { server in
            let store = makeStore(server)
            let scope = try scope()
            try await store.set(MemoryRecord(key: try MemoryKey(validating: "decisions/sync"),
                                             value: "Keep FooManager."), in: scope)
            try await store.set(MemoryRecord(key: try MemoryKey(validating: "tasks/one"),
                                             value: "unrelated"), in: scope)

            #expect(try await store.list(prefix: "decisions/", limit: 10, in: scope).count == 1)
            let hits = try await store.search(MemoryQuery(text: "FooManager"), in: scope)
            #expect(hits.first?.key.rawValue == "decisions/sync")
        }
    }

    @Test func appendExtendsAndDeleteRemovesFromTheIndex() async throws {
        try await withServer { server in
            let store = makeStore(server)
            let scope = try scope()
            let key = try MemoryKey(validating: "sessions/log")
            _ = try await store.append("first", to: key, in: scope)
            let second = try await store.append("second", to: key, in: scope)
            #expect(second.value == "first\nsecond")

            #expect(try await store.delete(key, in: scope))
            #expect(try await store.get(key, in: scope) == nil)
            #expect(try await store.list(prefix: "", limit: 10, in: scope).isEmpty)
            #expect(server.commandLog.contains { $0.first == "ZREM" })
        }
    }

    @Test func rewritePreservesCreationTime() async throws {
        try await withServer { server in
            let store = makeStore(server)
            let scope = try scope()
            let key = try MemoryKey(validating: "architecture/sync")
            let created = Date(timeIntervalSince1970: 1_000)
            try await store.set(MemoryRecord(key: key, value: "v1",
                                             createdAt: created, updatedAt: created), in: scope)
            try await store.set(MemoryRecord(key: key, value: "v2"), in: scope)

            let stored = try await store.get(key, in: scope)
            #expect(stored?.value == "v2")
            #expect(stored?.createdAt.timeIntervalSince1970 == 1_000)
            #expect((stored?.updatedAt ?? .distantPast) > created)
        }
    }

    @Test func oversizedValuesNeverReachTheServer() async throws {
        try await withServer { server in
            let store = makeStore(server, limits: MemoryLimits(maximumValueBytes: 64))
            let scope = try scope()
            await #expect(throws: MemoryError.self) {
                try await store.set(MemoryRecord(key: try MemoryKey(validating: "big"),
                                                 value: String(repeating: "x", count: 100)),
                                    in: scope)
            }
            #expect(!server.commandLog.contains { $0.first == "SET" })
        }
    }

    @Test func bootstrapStaysBounded() async throws {
        try await withServer { server in
            let limits = MemoryLimits(bootstrapRecords: 3, bootstrapBytes: 10_000)
            let store = makeStore(server, limits: limits)
            let scope = try scope()
            for index in 0..<12 {
                try await store.set(MemoryRecord(key: try MemoryKey(validating: "facts/f\(index)"),
                                                 value: "value \(index)",
                                                 importance: Double(index) / 12), in: scope)
            }

            let bootstrap = try await store.sessionInit(MemorySession(id: "s"), in: scope)
            #expect(bootstrap.records.count == 3)
            #expect(bootstrap.records.first?.importance ?? 0 > 0.8)
        }
    }

    @Test func appliesTheConfiguredMemoryCeiling() async throws {
        try await withServer { server in
            var configuration = server.configuration
            configuration.maximumMemoryBytes = 512 << 20
            let connection = ValkeyConnection(configuration: configuration)
            try await connection.connect()

            // A durable store must not evict: a full instance should refuse
            // writes rather than quietly drop facts the model relies on.
            #expect(server.configuredValues["maxmemory"] == String(512 << 20))
            #expect(server.configuredValues["maxmemory-policy"] == "noeviction")
            await connection.close()
        }
    }

    @Test func timesOutWhenTheServerNeverReplies() async throws {
        try await withServer { server in
            server.swallow("GET")
            let store = makeStore(server)
            let scope = try scope()

            let started = Date()
            await #expect(throws: MemoryError.self) {
                _ = try await store.get(try MemoryKey(validating: "facts/a"), in: scope)
            }
            // The deadline is the point: a memory read must not hold a
            // generation open indefinitely.
            #expect(Date().timeIntervalSince(started) < 3)
        }
    }

    @Test func surfacesServerErrorsAsBackendFailures() async throws {
        try await withServer { server in
            server.fail("SET", with: "OOM command not allowed when used memory > 'maxmemory'")
            let store = makeStore(server)

            await #expect(throws: MemoryError.self) {
                try await store.set(MemoryRecord(key: try MemoryKey(validating: "facts/a"),
                                                 value: "v"), in: try self.scope())
            }
        }
    }

    @Test func unavailableServerFailsWithoutHanging() async throws {
        // Nothing is listening on this port: the store must report a backend
        // failure quickly, which is what lets the server degrade instead of
        // stalling a request.
        var configuration = ValkeyConfiguration(host: "127.0.0.1", port: 1)
        configuration.connectTimeoutMilliseconds = 200
        configuration.operationTimeoutMilliseconds = 200
        let store = ValkeyMemoryStore(connection: ValkeyConnection(configuration: configuration))

        let started = Date()
        await #expect(throws: MemoryError.self) {
            _ = try await store.get(try MemoryKey(validating: "facts/a"), in: try self.scope())
        }
        #expect(Date().timeIntervalSince(started) < 5)
    }

    @Test func concurrentSessionsSharingOneConnectionStayConsistent() async throws {
        try await withServer { server in
            let store = makeStore(server)
            let scopeA = try scope("repo-a")
            let scopeB = try scope("repo-b")

            // Pipelining is the reason one connection is enough; if replies
            // were ever matched to the wrong request this is where it shows.
            await withTaskGroup(of: Void.self) { group in
                for index in 0..<20 {
                    group.addTask {
                        let scope = index.isMultiple(of: 2) ? scopeA : scopeB
                        guard let key = try? MemoryKey(validating: "concurrent/k\(index)") else {
                            return
                        }
                        try? await store.set(MemoryRecord(key: key, value: "\(index)"), in: scope)
                    }
                }
            }

            for index in 0..<20 {
                let scope = index.isMultiple(of: 2) ? scopeA : scopeB
                let key = try MemoryKey(validating: "concurrent/k\(index)")
                #expect(try await store.get(key, in: scope)?.value == "\(index)")
            }
        }
    }
}
