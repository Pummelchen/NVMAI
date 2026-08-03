import Darwin
import Foundation
import TurboFieldfareServerCore

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
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        promptCacheMode: arguments.promptCacheMode,
        promptCacheMaximumEntries: arguments.promptCacheMaximumEntries,
        promptCacheMemoryLimitBytes: arguments.promptCacheMemoryMiB * 1_048_576,
        promptCacheDiskDirectory: arguments.promptCacheDiskDirectory.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        promptCacheDiskLimitBytes: arguments.promptCacheDiskMiB * 1_048_576,
        prefillChunkTokens: arguments.prefillChunkTokens)
    let modelID = arguments.modelIDOverride ?? backend.defaultModelID
    let server = TurboFieldfareHTTPServer(
        modelID: modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        chatDialect: backend.chatDialect)
    _ = try await server.start(port: arguments.port)
    let diskCache = arguments.promptCacheDiskDirectory ?? "off"
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(modelID) context=\(arguments.maxContext) prefill_chunk=\(backend.prefillChunkTokens) prompt_cache=\(arguments.promptCacheMode.rawValue) prompt_cache_memory_mib=\(arguments.promptCacheMemoryMiB) prompt_cache_disk=\(diskCache)")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
