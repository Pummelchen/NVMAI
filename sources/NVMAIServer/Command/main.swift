import Darwin
import Foundation
import NVMAIMemory
import NVMAIServerCore

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    // Built lazily: the CPU engine serves an affine snapshot, which has no
    // manifest and would be rejected by a plan that expects an install.
    // Constructing it eagerly printed that rejection on every CPU launch.
    let makePlan = { ModelSessionPlan(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        promptCacheMode: arguments.promptCacheMode,
        promptCacheMaximumEntries: arguments.promptCacheMaximumEntries,
        promptCacheMemoryLimitBytes: arguments.promptCacheMemoryMiB * 1_048_576,
        promptCacheDiskDirectory: arguments.promptCacheDiskDirectory.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        promptCacheDiskLimitBytes: arguments.promptCacheDiskMiB * 1_048_576,
        prefillChunkTokens: arguments.prefillChunkTokens,
        kvCachePrecision: arguments.kvCachePrecision,
        ropeScalingMode: arguments.ropeScalingMode,
        thinkingMode: arguments.thinkingMode,
        reasoningEffort: arguments.reasoningEffort,
        expertCacheSlots: arguments.expertCacheSlots,
        expertCacheBudgetBytes: arguments.expertCacheBudgetBytes,
        mtpModelDirectory: arguments.mtpModel.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        mtpMemoryMiB: arguments.mtpMemoryMiB) }

    let backend: any ServerInferenceBackend
    let facts: ModelSessionFacts
    var managed: ManagedModelBackend?
    // The reasoning profile comes from an install's manifest, which a CPU
    // snapshot does not have. These models carry no reasoning-effort control
    // either, so the profile is the family's plain default.
    var reasoningProfile: ServerReasoningProfile?

    if arguments.cpu {
        // A different engine entirely: no Metal context, no expert
        // streaming, no prompt cache. Everything above it -- both API
        // surfaces, the memory subsystem, the watchdogs -- is unchanged,
        // which is the point of putting it behind the same protocol.
        let directory = URL(fileURLWithPath: arguments.model).standardizedFileURL
        let started = ContinuousClock.now
        let cpuBackend = try await CPUModelBackend(
            snapshotDirectory: directory,
            maximumContext: arguments.maxContext,
            resident: arguments.cpuResident)
        backend = cpuBackend
        let elapsed = started.duration(to: .now)
        let seconds = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        let identifier = arguments.modelIDOverride
            ?? directory.lastPathComponent
        facts = ModelSessionFacts(
            modelID: identifier,
            prefillChunkTokens: 0,
            promptCacheMode: .off,
            expertCacheSlots: 0)
        print(String(format: "CPU engine: %@ %@in %.1fs, %d threads",
                     identifier,
                     cpuBackend.residentBytes > 0
                        ? "\(cpuBackend.residentBytes / 1_000_000) MB resident, " : "",
                     seconds, cpuBackend.threads))
        reasoningProfile = ServerReasoningProfile(
            family: .qwen36,
            thinkingMode: arguments.thinkingMode,
            reasoningEffort: nil)
    } else if arguments.managesResidency {
        let plan = makePlan()
        // Reads manifest.json only; a bad --model still fails here at launch
        // rather than on the first request.
        facts = try plan.previewFacts(modelIDOverride: arguments.modelIDOverride)
        let residency = ManagedModelBackend(
            plan: plan,
            facts: facts,
            idleTimeout: arguments.idleUnloadSeconds > 0
                ? .seconds(arguments.idleUnloadSeconds) : nil)
        managed = residency
        backend = residency
    } else {
        let plan = makePlan()
        let session = try await plan.makeSession()
        backend = session
        facts = ModelSessionFacts(
            modelID: arguments.modelIDOverride ?? session.defaultModelID,
            prefillChunkTokens: session.prefillChunkTokens,
            promptCacheMode: session.promptCacheMode,
            expertCacheSlots: session.expertCacheSlots)
    }

    // Persistent memory wraps whatever backend was built: one decorator on
    // the way in, and nothing at all when it is disabled.
    let servingBackend = ServerMemoryFactory.wrap(backend)

    let server = NVMAIHTTPServer(
        modelID: facts.modelID,
        queueLimit: arguments.queueLimit,
        backend: servingBackend,
        reasoningProfile: try reasoningProfile ?? makePlan().reasoningProfile())
    _ = try await server.start(port: arguments.port)
    let diskCache = facts.promptCacheMode == .off
        ? "off" : arguments.promptCacheDiskDirectory ?? "off"
    let cacheMemoryMiB = facts.promptCacheMode == .off
        ? 0 : arguments.promptCacheMemoryMiB
    let mtp = arguments.mtpModel == nil ? "off" : "on:\(arguments.mtpMemoryMiB)MiB"
    let residencyBanner = arguments.managesResidency
        ? " lazy_load=on idle_unload=\(arguments.idleUnloadSeconds > 0 ? "\(arguments.idleUnloadSeconds)s" : "off")"
        : ""
    print("NVMAIServer ready at http://127.0.0.1:\(arguments.port) model=\(facts.modelID) context=\(arguments.maxContext) prefill_chunk=\(facts.prefillChunkTokens)\(facts.expertCacheSlots > 0 ? " expert_slots=\(facts.expertCacheSlots)" : "") prompt_cache=\(facts.promptCacheMode.rawValue) prompt_cache_memory_mib=\(cacheMemoryMiB) prompt_cache_disk=\(diskCache) thinking=\(arguments.thinkingMode.rawValue) mtp=\(mtp)\(residencyBanner)")
    WatchdogConfiguration.shared.announce()
    if arguments.unloadDiscardsWarmCache {
        FileHandle.standardError.write(Data(
            ("warning: --idle-unload-seconds drops the in-memory prompt cache with "
                + "the model; add --prompt-cache-disk <dir> so entries survive an "
                + "unload, or the first request after each unload pays a full "
                + "cold prefill\n").utf8))
    }

    _ = await signals.wait()
    try await server.shutdown()
    // After the server, so nothing is still writing: this flushes memory that
    // has not reached a session boundary and releases the workspace lock.
    if let memory = servingBackend as? MemoryBackend {
        await memory.shutDown()
    }
    // After the server, so the reaper cannot outlive it.
    await managed?.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
