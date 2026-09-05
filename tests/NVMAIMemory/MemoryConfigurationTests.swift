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

    @Test func toolSurfaceIsChosenByName() {
        func surface(_ value: String) -> MemoryToolSurface {
            MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY": "1",
                                                 "NVMAI_MEMORY_TOOLS": value]).toolSurface
        }
        #expect(MemoryConfiguration().toolSurface == .off)
        #expect(surface("off") == .off)
        #expect(surface("minimal") == .minimal)
        #expect(surface("full") == .full)
        // The old boolean spelling still works.
        #expect(surface("1") == .full)
        #expect(surface("0") == .off)
        // Minimal is the write plus the targeted read; discovery comes from
        // the bootstrap, which already lists what exists.
        #expect(MemoryToolSurface.minimal.toolNames == ["memory_set", "memory_get"])
        #expect(MemoryToolSurface.full.toolNames == MemoryTools.names)
        #expect(MemoryTools.definitions(surface: .minimal).count == 2)
        #expect(MemoryTools.definitions(surface: .off).isEmpty)
    }

    @Test func memoryIsOffUnlessAskedFor() {
        // Nothing about serving changes for someone who has not enabled it.
        #expect(!MemoryConfiguration.fromEnvironment([:]).isEnabled)
        #expect(!MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY_DIR": "/tmp/x"]).isEnabled)
        #expect(MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY": "1"]).isEnabled)
        #expect(MemoryConfiguration.fromEnvironment(["NVMAI_MEMORY": "on"]).isEnabled)
    }

    @Test func storageLandsOneFilePerWorkspace() throws {
        var storage = ContinuityStorageConfiguration(
            directory: URL(fileURLWithPath: "/tmp/nvmai-memory"))
        let scope = try MemoryScope(namespace: "nvmai", user: "ada", workspace: "repo-1234abcd")
        #expect(storage.journalURL(for: scope).path
                == "/tmp/nvmai-memory/nvmai/ada/repo-1234abcd.ndjson")
        // Two workspaces never share a file, which is what makes deleting one
        // project's memory a file removal rather than an edit.
        let other = try MemoryScope(namespace: "nvmai", user: "ada", workspace: "repo-5678ef00")
        #expect(storage.journalURL(for: scope) != storage.journalURL(for: other))
        storage.directory = URL(fileURLWithPath: "/var/lib/nvmai")
        #expect(storage.journalURL(for: scope).path.hasPrefix("/var/lib/nvmai/"))
    }

    @Test func environmentOverridesEveryDocumentedKnob() {
        let configuration = MemoryConfiguration.fromEnvironment([
            "NVMAI_MEMORY": "1",
            "NVMAI_MEMORY_DIR": "/tmp/nvmai-memory-test",
            "NVMAI_MEMORY_FSYNC": "1",
            "NVMAI_MEMORY_CACHE_MIB": "128",
            "NVMAI_MEMORY_NAMESPACE": "team",
            "NVMAI_MEMORY_USER": "ada",
            "NVMAI_MEMORY_WORKSPACE": "explicit-workspace",
            "NVMAI_MEMORY_MAX_VALUE_BYTES": "2048",
            "NVMAI_MEMORY_BOOTSTRAP_LIMIT": "5",
            "NVMAI_MEMORY_TOOL_ROUNDS": "2",
            "NVMAI_MEMORY_TOOLS": "minimal",
            "NVMAI_MEMORY_CONSOLIDATION": "1",
        ])
        #expect(configuration.storage.directory.path == "/tmp/nvmai-memory-test")
        #expect(configuration.storage.synchronizesEveryWrite)
        #expect(configuration.storage.maximumMemoryBytes == 128 << 20)
        #expect(configuration.namespace == "team")
        #expect(configuration.user == "ada")
        #expect(configuration.workspace == "explicit-workspace")
        #expect(configuration.limits.maximumValueBytes == 2048)
        #expect(configuration.limits.bootstrapRecords == 5)
        #expect(configuration.maximumToolRounds == 2)
        #expect(configuration.toolSurface == .minimal)
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

    @Test func summaryNamesTheSettings() {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.storage.maximumMemoryBytes = 512 << 20
        let summary = configuration.summary
        #expect(summary.contains("memory enabled=true"))
        #expect(summary.contains("store=in-process"))
        #expect(summary.contains("cache=512MiB"))
        // There is no host and no port to report any more.
        #expect(!summary.contains("6379"))
    }

    /// The byte budget the machine is sized for has to become a real bound,
    /// or it is decoration. It becomes a worst-case item ceiling.
    @Test func theCacheBudgetBecomesAnItemCeiling() {
        var configuration = MemoryConfiguration()
        configuration.storage.maximumMemoryBytes = 256 << 20
        configuration.limits.maximumValueBytes = 64 << 10
        #expect(MemoryService.itemCeiling(for: configuration) == 4096)

        configuration.storage.maximumMemoryBytes = 1 << 30
        #expect(MemoryService.itemCeiling(for: configuration) == 16384)

        // A tiny budget still leaves room to be useful.
        configuration.storage.maximumMemoryBytes = 1024
        #expect(MemoryService.itemCeiling(for: configuration) == 64)
    }
}
