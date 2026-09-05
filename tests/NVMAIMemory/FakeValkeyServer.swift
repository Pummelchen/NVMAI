import Foundation
import NIOCore
import NIOPosix
@testable import NVMAIMemory

/// A Valkey stand-in that speaks RESP over a real socket.
///
/// The point is to test the client's wire behaviour without a server on the
/// machine or in CI: framing across packet boundaries, pipelining, error
/// replies, and what happens when a reply never comes. It implements only the
/// commands `ValkeyMemoryStore` issues, and it is deliberately simple enough
/// to be obviously correct.
final class FakeValkeyServer: @unchecked Sendable {
    /// unchecked-invariant: `state` is the only mutable member and every
    /// access goes through `lock`. The channel handler runs on event loops,
    /// tests read counters from their own task, so the lock is the boundary.
    private final class State {
        var strings: [String: String] = [:]
        var sortedSets: [String: [(member: String, score: Double)]] = [:]
        var lists: [String: [String]] = [:]
        var commandLog: [[String]] = []
        var configSets: [String: String] = [:]
        /// Commands that get no reply at all, for timeout tests.
        var swallowedCommands: Set<String> = []
        /// Commands answered with an error reply.
        var failingCommands: [String: String] = [:]
    }

    private let state = State()
    private let lock = NSLock()
    private let group: MultiThreadedEventLoopGroup
    private var channel: Channel?
    private(set) var port: Int = 0

    init() {
        group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    }

    func start() throws {
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(.backlog, value: 8)
            .serverChannelOption(.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { [weak self] channel in
                guard let self else { return channel.close() }
                return channel.pipeline.addHandler(Handler(server: self))
            }
        let channel = try bootstrap.bind(host: "127.0.0.1", port: 0).wait()
        self.channel = channel
        self.port = channel.localAddress?.port ?? 0
    }

    func stop() {
        try? channel?.close().wait()
        try? group.syncShutdownGracefully()
    }

    var configuration: ValkeyConfiguration {
        ValkeyConfiguration(host: "127.0.0.1", port: port,
                            connectTimeoutMilliseconds: 1_000,
                            operationTimeoutMilliseconds: 300)
    }

    // MARK: - Test controls

    func swallow(_ command: String) {
        lock.withLock { _ = state.swallowedCommands.insert(command.uppercased()) }
    }

    func fail(_ command: String, with message: String) {
        lock.withLock { state.failingCommands[command.uppercased()] = message }
    }

    var commandLog: [[String]] { lock.withLock { state.commandLog } }

    var configuredValues: [String: String] { lock.withLock { state.configSets } }

    func storedKeys() -> [String] { lock.withLock { Array(state.strings.keys).sorted() } }

    // MARK: - Command execution

    fileprivate func execute(_ arguments: [String]) -> RESPValue? {
        lock.withLock {
            state.commandLog.append(arguments)
            guard let name = arguments.first?.uppercased() else { return .error("ERR empty") }
            if state.swallowedCommands.contains(name) { return nil }
            if let message = state.failingCommands[name] { return .error(message) }

            switch name {
            case "AUTH", "SELECT", "PING":
                return .simpleString("OK")
            case "CONFIG":
                if arguments.count >= 4, arguments[1].uppercased() == "SET" {
                    state.configSets[arguments[2].lowercased()] = arguments[3]
                }
                return .simpleString("OK")
            case "SET":
                guard arguments.count >= 3 else { return .error("ERR wrong number of arguments") }
                state.strings[arguments[1]] = arguments[2]
                return .simpleString("OK")
            case "GET":
                guard let value = state.strings[arguments[1]] else { return .bulkString(nil) }
                return .bulkString(buffer(value))
            case "MGET":
                let values = arguments.dropFirst().map { key -> RESPValue in
                    state.strings[key].map { .bulkString(buffer($0)) } ?? .bulkString(nil)
                }
                return .array(Array(values))
            case "DEL":
                var removed = 0
                for key in arguments.dropFirst() where state.strings.removeValue(forKey: key) != nil {
                    removed += 1
                }
                return .integer(removed)
            case "EXISTS":
                return .integer(state.strings[arguments[1]] != nil ? 1 : 0)
            case "ZADD":
                guard arguments.count >= 4, let score = Double(arguments[2]) else {
                    return .error("ERR syntax")
                }
                var entries = state.sortedSets[arguments[1]] ?? []
                entries.removeAll { $0.member == arguments[3] }
                entries.append((member: arguments[3], score: score))
                state.sortedSets[arguments[1]] = entries
                return .integer(1)
            case "ZREM":
                var entries = state.sortedSets[arguments[1]] ?? []
                let before = entries.count
                entries.removeAll { $0.member == arguments[2] }
                state.sortedSets[arguments[1]] = entries
                return .integer(before - entries.count)
            case "ZRANGE":
                let entries = state.sortedSets[arguments[1]] ?? []
                let reversed = arguments.contains { $0.uppercased() == "REV" }
                let sorted = entries.sorted { reversed ? $0.score > $1.score : $0.score < $1.score }
                let start = Int(arguments[2]) ?? 0
                let stop = Int(arguments[3]) ?? -1
                let upper = stop < 0 ? sorted.count - 1 : min(stop, sorted.count - 1)
                guard start <= upper, start < sorted.count else { return .array([]) }
                let slice = sorted[start...upper].map { RESPValue.bulkString(buffer($0.member)) }
                return .array(Array(slice))
            case "LPUSH":
                var list = state.lists[arguments[1]] ?? []
                list.insert(contentsOf: arguments.dropFirst(2).reversed(), at: 0)
                state.lists[arguments[1]] = list
                return .integer(list.count)
            case "LTRIM":
                var list = state.lists[arguments[1]] ?? []
                let start = Int(arguments[2]) ?? 0
                let stop = Int(arguments[3]) ?? -1
                let upper = stop < 0 ? list.count - 1 : min(stop, list.count - 1)
                list = start <= upper && start < list.count ? Array(list[start...upper]) : []
                state.lists[arguments[1]] = list
                return .simpleString("OK")
            case "LRANGE":
                let list = state.lists[arguments[1]] ?? []
                return .array(list.map { .bulkString(buffer($0)) })
            default:
                return .error("ERR unknown command '\(name)'")
            }
        }
    }

    private func buffer(_ text: String) -> ByteBuffer {
        var buffer = ByteBufferAllocator().buffer(capacity: text.utf8.count)
        buffer.writeString(text)
        return buffer
    }

    /// Parses requests and writes replies. Requests are RESP arrays of bulk
    /// strings, which is all a client sends.
    /// unchecked-invariant: `accumulator` is touched only on the child
    /// channel's event loop; the server it points at is internally locked.
    private final class Handler: ChannelInboundHandler, @unchecked Sendable {
        typealias InboundIn = ByteBuffer
        typealias OutboundOut = ByteBuffer

        private let server: FakeValkeyServer
        private var accumulator: ByteBuffer?

        init(server: FakeValkeyServer) { self.server = server }

        func channelRead(context: ChannelHandlerContext, data: NIOAny) {
            var incoming = unwrapInboundIn(data)
            if accumulator == nil {
                accumulator = incoming
            } else {
                accumulator?.writeBuffer(&incoming)
            }
            guard var buffer = accumulator else { return }
            defer {
                buffer.discardReadBytes()
                accumulator = buffer.readableBytes > 0 ? buffer : nil
            }
            while true {
                // `try?` flattens the parser's optional, so this covers
                // both "needs more bytes" and a malformed request; the fake
                // simply waits in either case.
                guard let value = try? RESPParser.parse(&buffer) else { return }
                guard let arguments = value.arrayValue?.compactMap(\.stringValue) else { continue }
                guard let reply = server.execute(arguments) else { continue }
                var out = context.channel.allocator.buffer(capacity: 32)
                Self.encode(reply, into: &out)
                context.writeAndFlush(wrapOutboundOut(out), promise: nil)
            }
        }

        static func encode(_ value: RESPValue, into buffer: inout ByteBuffer) {
            switch value {
            case .simpleString(let text): buffer.writeString("+\(text)\r\n")
            case .error(let text): buffer.writeString("-\(text)\r\n")
            case .integer(let number): buffer.writeString(":\(number)\r\n")
            case .bulkString(let payload):
                guard var payload else {
                    buffer.writeString("$-1\r\n")
                    return
                }
                buffer.writeString("$\(payload.readableBytes)\r\n")
                buffer.writeBuffer(&payload)
                buffer.writeString("\r\n")
            case .array(let items):
                guard let items else {
                    buffer.writeString("*-1\r\n")
                    return
                }
                buffer.writeString("*\(items.count)\r\n")
                for item in items { encode(item, into: &buffer) }
            }
        }
    }
}
