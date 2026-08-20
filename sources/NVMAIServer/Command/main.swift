import Darwin
import Foundation
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
    let plan = ModelSessionPlan(
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
        expertCacheSlots: arguments.expertCacheSlots,
        expertCacheBudgetBytes: arguments.expertCacheBudgetBytes,
        mtpModelDirectory: arguments.mtpModel.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        mtpMemoryMiB: arguments.mtpMemoryMiB)

    let backend: any ServerInferenceBackend
    let facts: ModelSessionFacts
    var managed: ManagedModelBackend?

    if arguments.managesResidency {
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
        let session = try await plan.makeSession()
        backend = session
        facts = ModelSessionFacts(
            modelID: arguments.modelIDOverride ?? session.defaultModelID,
            prefillChunkTokens: session.prefillChunkTokens,
            promptCacheMode: session.promptCacheMode,
            expertCacheSlots: session.expertCacheSlots)
    }

    let server = NVMAIHTTPServer(
        modelID: facts.modelID,
        queueLimit: arguments.queueLimit,
        backend: backend)
    _ = try await server.start(port: arguments.port)
    let diskCache = facts.promptCacheMode == .off
        ? "off" : arguments.promptCacheDiskDirectory ?? "off"
    let cacheMemoryMiB = facts.promptCacheMode == .off
        ? 0 : arguments.promptCacheMemoryMiB
    let mtp = arguments.mtpModel == nil ? "off" : "on:\(arguments.mtpMemoryMiB)MiB"
    let residencyBanner = arguments.managesResidency
        ? " lazy_load=on idle_unload=\(arguments.idleUnloadSeconds > 0 ? "\(arguments.idleUnloadSeconds)s" : "off")"
        : ""
    print("NVMAIServer ready at http://127.0.0.1:\(arguments.port) model=\(facts.modelID) context=\(arguments.maxContext) prefill_chunk=\(facts.prefillChunkTokens)\(facts.expertCacheSlots > 0 ? " expert_slots=\(facts.expertCacheSlots)" : "") prompt_cache=\(facts.promptCacheMode.rawValue) prompt_cache_memory_mib=\(cacheMemoryMiB) prompt_cache_disk=\(diskCache) mtp=\(mtp)\(residencyBanner)")
    if arguments.unloadDiscardsWarmCache {
        FileHandle.standardError.write(Data(
            ("warning: --idle-unload-seconds drops the in-memory prompt cache with "
                + "the model; add --prompt-cache-disk <dir> so entries survive an "
                + "unload, or the first request after each unload pays a full "
                + "cold prefill\n").utf8))
    }

    _ = await signals.wait()
    try await server.shutdown()
    // After the server, so the reaper cannot outlive it.
    await managed?.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
