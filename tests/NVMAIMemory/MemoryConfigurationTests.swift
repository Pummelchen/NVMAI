import Foundation
import Testing
@testable import NVMAIMemory

@Suite struct MemoryConfigurationTests {
    @Test func cacheCeilingFollowsMachineMemory() {
        let gigabyte = UInt64(1) << 30
        // The deployment rule: a memory store is worth a fixed slice, not a
        // fraction, because its working set is a few thousand short facts.
        #expect(MemoryConfiguration.defaultCacheBytes(physicalMemory: 8 * gigabyte) == 256 << 20)
        #expect(MemoryConfiguration.defaultCacheBytes(physicalMemory: 16 * gigabyte) == 512 << 20)
        #expect(MemoryConfiguration.defaultCacheBytes(physicalMemory: 24 * gigabyte) == 1 << 30)
        #expect(MemoryConfiguration.defaultCacheBytes(physicalMemory: 128 * gigabyte) == 1 << 30)
        // Boundaries land on the lower tier, and a tiny machine is not given
        // more than the smallest.
        #expect(MemoryConfiguration.defaultCacheBytes(physicalMemory: 4 * gigabyte) == 256 << 20)
        #expect(MemoryConfiguration.defaultCacheBytes(
            physicalMemory: 8 * gigabyte + 1) == 512 << 20)
    }

    @Test func memoryIsOffUnlessAskedFor() {
        // Nothing about serving changes for someone who has not enabled it.
        #expect(!MemoryConfiguration.fromEnvironment([:]).isEnabled)
        #expect(!MemoryConfiguration.fromEnvironment(["VALKEY_URL": "redis://host:1"]).isEnabled)
        #expect(MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY": "1"]).isEnabled)
        #expect(MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY": "on"]).isEnabled)
    }

    @Test func parsesAValkeyURL() throws {
        let configuration = try #require(
            ValkeyConfiguration.parse(url: "redis://ada:secret@db.local:6380/3"))
        #expect(configuration.host == "db.local")
        #expect(configuration.port == 6380)
        #expect(configuration.username == "ada")
        #expect(configuration.password == "secret")
        #expect(configuration.database == 3)

        #expect(ValkeyConfiguration.parse(url: "valkey://127.0.0.1")?.port == 6379)
        #expect(ValkeyConfiguration.parse(url: "https://example.com") == nil)
        #expect(ValkeyConfiguration.parse(url: "not a url") == nil)
    }

    @Test func environmentOverridesEveryDocumentedKnob() {
        let configuration = MemoryConfiguration.fromEnvironment([
            "NVMAI_MEMORY": "1",
            "VALKEY_URL": "redis://127.0.0.1:6390/2",
            "VALKEY_PASSWORD": "from-env",
            "NVMAI_MEMORY_TIMEOUT_MS": "75",
            "NVMAI_MEMORY_CACHE_MIB": "128",
            "NVMAI_MEMORY_NAMESPACE": "team",
            "NVMAI_MEMORY_USER": "ada",
            "NVMAI_MEMORY_WORKSPACE": "explicit-workspace",
            "NVMAI_MEMORY_MAX_VALUE_BYTES": "2048",
            "NVMAI_MEMORY_BOOTSTRAP_LIMIT": "5",
            "NVMAI_MEMORY_TOOL_ROUNDS": "2",
            "NVMAI_MEMORY_TOOLS": "0",
            "NVMAI_MEMORY_CONSOLIDATION": "1",
        ])
        #expect(configuration.valkey.port == 6390)
        #expect(configuration.valkey.database == 2)
        // An explicit password beats the one in the URL.
        #expect(configuration.valkey.password == "from-env")
        #expect(configuration.valkey.operationTimeoutMilliseconds == 75)
        #expect(configuration.valkey.maximumMemoryBytes == 128 << 20)
        #expect(configuration.namespace == "team")
        #expect(configuration.user == "ada")
        #expect(configuration.workspace == "explicit-workspace")
        #expect(configuration.limits.maximumValueBytes == 2048)
        #expect(configuration.limits.bootstrapRecords == 5)
        #expect(configuration.maximumToolRounds == 2)
        #expect(!configuration.exposesTools)
        #expect(configuration.sessionConsolidation)
    }

    @Test func workspaceComesFromTheLaunchDirectoryWhenNotNamed() {
        let configuration = MemoryConfiguration.fromEnvironment([
            "NVMAI_MEMORY": "1",
            "NVMAI_WORKSPACE_DIR": "/Users/ada/src/nvmai",
        ])
        #expect(configuration.workspace.hasPrefix("nvmai-"))
        #expect(configuration.scope() != nil)
    }

    @Test func twoCheckoutsOfOneRepositoryGetDifferentWorkspaces() {
        // Same directory name, different paths: sharing memory between them
        // would be the cross-project leak the scoping exists to prevent.
        let first = MemoryConfiguration.workspaceIdentifier(forPath: "/Users/ada/a/nvmai")
        let second = MemoryConfiguration.workspaceIdentifier(forPath: "/Users/ada/b/nvmai")
        #expect(first != second)
        #expect(first.hasPrefix("nvmai-") && second.hasPrefix("nvmai-"))
        // Stable across processes: a restart must land on the same memory.
        #expect(first == MemoryConfiguration.workspaceIdentifier(forPath: "/Users/ada/a/nvmai/"))
    }

    @Test func perRequestWorkspaceCanBeRefused() throws {
        var configuration = MemoryConfiguration()
        configuration.workspace = "pinned"
        configuration.allowsPerRequestWorkspace = false
        #expect(configuration.scope(workspaceOverride: "other")?.workspace == "pinned")

        configuration.allowsPerRequestWorkspace = true
        #expect(configuration.scope(workspaceOverride: "other")?.workspace == "other")
        // An unusable override yields no scope at all rather than silently
        // falling back to a shared one.
        #expect(configuration.scope(workspaceOverride: "../escape") == nil)
    }

    @Test func summaryNamesTheSettingsAndNoCredential() {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.valkey.password = "hunter2"
        configuration.valkey.username = "ada"
        let summary = configuration.summary
        #expect(summary.contains("memory enabled=true"))
        #expect(summary.contains("127.0.0.1:6379"))
        #expect(!summary.contains("hunter2"))
        #expect(!summary.contains("ada"))
    }
}
