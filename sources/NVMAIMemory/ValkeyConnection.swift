import Foundation
import NIOCore
import NIOPosix

/// A single pipelined connection to Valkey.
///
/// One connection, reused for the life of the server, is the right shape
/// here: memory operations are small and infrequent next to inference, and
/// reconnecting per operation is the cost the requirements call out. Commands
/// are pipelined in FIFO order, which is what makes a single connection
/// enough for concurrent sessions.
///
/// The actor owns the channel; nothing outside it touches NIO.
public actor ValkeyConnection {
    private let group: EventLoopGroup
    private let ownsGroup: Bool
    private let configuration: ValkeyConfiguration
    private var channel: Channel?

    public init(configuration: ValkeyConfiguration, group: EventLoopGroup? = nil) {
        self.configuration = configuration
        if let group {
            self.group = group
            self.ownsGroup = false
        } else {
            self.group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
            self.ownsGroup = true
        }
    }

    deinit {
        if ownsGroup { try? group.syncShutdownGracefully() }
    }

    public var isConnected: Bool { channel?.isActive == true }

    /// Opens the connection and completes the handshake (auth, database
    /// selection, and the memory ceiling when one is configured).
    public func connect() async throws {
        if channel?.isActive == true { return }
        let bootstrap = ClientBootstrap(group: group)
            .channelOption(.socketOption(.so_reuseaddr), value: 1)
            .connectTimeout(.milliseconds(Int64(configuration.connectTimeoutMilliseconds)))
            .channelInitializer { channel in
                channel.pipeline.addHandler(ResponseHandler())
            }
        do {
            let channel = try await bootstrap.connect(host: configuration.host,
                                                      port: configuration.port).get()
            self.channel = channel
        } catch {
            throw ValkeyError.connectionFailed("\(configuration.host):\(configuration.port): \(error)")
        }
        try await handshake()
    }

    private func handshake() async throws {
        if let password = configuration.password, !password.isEmpty {
            let command = configuration.username.map { ["AUTH", $0, password] } ?? ["AUTH", password]
            _ = try await send(command)
        }
        if configuration.database != 0 {
            _ = try await send(["SELECT", String(configuration.database)])
        }
        // The memory ceiling is applied here rather than left to the operator
        // because a store meant to be durable must not evict: `noeviction`
        // makes a full instance fail writes loudly instead of silently
        // dropping the facts the model relies on.
        if let bytes = configuration.maximumMemoryBytes {
            _ = try? await send(["CONFIG", "SET", "maxmemory", String(bytes)])
            _ = try? await send(["CONFIG", "SET", "maxmemory-policy", "noeviction"])
        }
    }

    public func close() async {
        let channel = self.channel
        self.channel = nil
        try? await channel?.close()
    }

    /// Sends one command and waits for its reply, subject to the operation
    /// timeout.
    ///
    /// The timeout is scheduled on the channel's event loop rather than raced
    /// from another task: the reply and the deadline then resolve in the same
    /// place, so a late reply cannot arrive after the caller gave up and be
    /// handed to the next command. A timeout closes the connection, because a
    /// pipelined connection whose replies are off by one would return another
    /// session's data.
    @discardableResult
    public func send(_ arguments: [String]) async throws -> RESPValue {
        guard let channel, channel.isActive else { throw ValkeyError.notConnected }
        var buffer = channel.allocator.buffer(capacity: 64)
        RESPEncoder.encode(arguments, into: &buffer)
        let deadline = TimeAmount.milliseconds(Int64(configuration.operationTimeoutMilliseconds))
        let label = arguments.first ?? "command"
        do {
            let value: RESPValue = try await withCheckedThrowingContinuation { continuation in
                let request = WriteRequest(buffer: buffer,
                                           continuation: continuation,
                                           timeout: deadline,
                                           label: label)
                channel.writeAndFlush(request, promise: nil)
            }
            if let message = value.errorMessage { throw ValkeyError.commandFailed(message) }
            return value
        } catch let error as ValkeyError {
            if case .timedOut = error { await close() }
            throw error
        }
    }
}

/// What the channel handler receives: the encoded command, the continuation
/// its reply resolves, and how long the reply may take.
///
/// unchecked-invariant: created on the connection actor and handed to the
/// channel exactly once. From the write onwards only the event loop touches
/// it, and the continuation is resumed exactly once, by whichever of the
/// reply, the timeout or channel teardown comes first.
struct WriteRequest: @unchecked Sendable {
    let buffer: ByteBuffer
    let continuation: CheckedContinuation<RESPValue, Error>
    let timeout: TimeAmount
    let label: String
}

/// Matches replies to requests in order.
///
/// unchecked-invariant: every mutable member is touched only on the channel's
/// event loop, which is where `write`, `channelRead`, the scheduled timeouts
/// and the teardown callbacks all run.
final class ResponseHandler: ChannelDuplexHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer
    typealias OutboundIn = WriteRequest
    typealias OutboundOut = ByteBuffer

    /// A request awaiting its reply. `isResolved` is what makes the
    /// three-way race (reply, timeout, disconnect) resume the continuation
    /// exactly once.
    ///
    /// unchecked-invariant: created, mutated and resolved only on the
    /// channel's event loop; the scheduled timeout runs on that same loop.
    private final class Pending: @unchecked Sendable {
        let continuation: CheckedContinuation<RESPValue, Error>
        let label: String
        var timeoutTask: Scheduled<Void>?
        var isResolved = false

        init(continuation: CheckedContinuation<RESPValue, Error>, label: String) {
            self.continuation = continuation
            self.label = label
        }

        func resolve(_ result: Result<RESPValue, Error>) {
            guard !isResolved else { return }
            isResolved = true
            timeoutTask?.cancel()
            timeoutTask = nil
            continuation.resume(with: result)
        }
    }

    private var pending: [Pending] = []
    private var accumulator: ByteBuffer?

    func write(context: ChannelHandlerContext, data: NIOAny, promise: EventLoopPromise<Void>?) {
        let request = unwrapOutboundIn(data)
        let entry = Pending(continuation: request.continuation, label: request.label)
        pending.append(entry)
        // Capture the channel, not the context: a context is only valid
        // inside a handler callback, while this closure runs later from the
        // loop's timer.
        let channel = context.channel
        entry.timeoutTask = context.eventLoop.scheduleTask(in: request.timeout) { [weak self] in
            guard let self, !entry.isResolved else { return }
            entry.resolve(.failure(ValkeyError.timedOut(entry.label)))
            self.pending.removeAll { $0 === entry }
            channel.close(promise: nil)
        }
        context.writeAndFlush(wrapOutboundOut(request.buffer), promise: promise)
    }

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
            do {
                guard let value = try RESPParser.parse(&buffer) else { return }
                deliver(.success(value))
            } catch {
                deliver(.failure(error))
                failAll(error)
                context.close(promise: nil)
                return
            }
        }
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        failAll(error)
        context.close(promise: nil)
    }

    func channelInactive(context: ChannelHandlerContext) {
        failAll(ValkeyError.notConnected)
        context.fireChannelInactive()
    }

    private func deliver(_ result: Result<RESPValue, Error>) {
        guard !pending.isEmpty else { return }
        let entry = pending.removeFirst()
        entry.resolve(result)
    }

    private func failAll(_ error: Error) {
        let waiting = pending
        pending.removeAll()
        for entry in waiting { entry.resolve(.failure(error)) }
    }
}
