import Foundation
import NIOCore

/// One RESP2 value. Valkey speaks Redis's wire protocol, and the subset this
/// store needs is small enough that a client is cheaper than a dependency:
/// the package has two, and adding a third to issue eight commands would be
/// the larger cost. `ValkeyMemoryStore` is the only caller, so replacing this
/// with a library later touches one file.
public enum RESPValue: Equatable, Sendable {
    case simpleString(String)
    case error(String)
    case integer(Int)
    case bulkString(ByteBuffer?)
    case array([RESPValue]?)

    public var stringValue: String? {
        switch self {
        case .simpleString(let value): return value
        case .bulkString(let buffer):
            guard var buffer else { return nil }
            return buffer.readString(length: buffer.readableBytes)
        case .integer(let value): return String(value)
        case .error, .array: return nil
        }
    }

    public var integerValue: Int? {
        switch self {
        case .integer(let value): return value
        case .simpleString, .bulkString: return stringValue.flatMap(Int.init)
        case .error, .array: return nil
        }
    }

    public var arrayValue: [RESPValue]? {
        if case .array(let items) = self { return items }
        return nil
    }

    public var errorMessage: String? {
        if case .error(let message) = self { return message }
        return nil
    }
}

/// Encodes a command as a RESP array of bulk strings, which is the only
/// request form a server needs to accept.
enum RESPEncoder {
    static func encode(_ arguments: [String], into buffer: inout ByteBuffer) {
        buffer.writeString("*\(arguments.count)\r\n")
        for argument in arguments {
            let bytes = argument.utf8
            buffer.writeString("$\(bytes.count)\r\n")
            buffer.writeString(argument)
            buffer.writeString("\r\n")
        }
    }
}

/// Incremental RESP2 parser.
///
/// Returns nil when the buffer holds only part of a value and leaves the
/// reader index untouched, so the caller can wait for more bytes and retry.
/// That is the whole contract: a decoder that consumed a partial value would
/// desynchronise the connection permanently.
enum RESPParser {
    static func parse(_ buffer: inout ByteBuffer) throws -> RESPValue? {
        let start = buffer.readerIndex
        guard let value = try parseValue(&buffer) else {
            buffer.moveReaderIndex(to: start)
            return nil
        }
        return value
    }

    private static func parseValue(_ buffer: inout ByteBuffer) throws -> RESPValue? {
        guard let marker = buffer.readInteger(as: UInt8.self) else { return nil }
        switch marker {
        case UInt8(ascii: "+"):
            return try readLine(&buffer).map { .simpleString($0) }
        case UInt8(ascii: "-"):
            return try readLine(&buffer).map { .error($0) }
        case UInt8(ascii: ":"):
            guard let line = try readLine(&buffer) else { return nil }
            guard let number = Int(line) else { throw ValkeyError.protocolViolation("integer '\(line)'") }
            return .integer(number)
        case UInt8(ascii: "$"):
            guard let line = try readLine(&buffer) else { return nil }
            guard let count = Int(line) else { throw ValkeyError.protocolViolation("bulk length '\(line)'") }
            if count < 0 { return .bulkString(nil) }
            guard buffer.readableBytes >= count + 2 else { return nil }
            let slice = buffer.readSlice(length: count)
            buffer.moveReaderIndex(forwardBy: 2)
            return .bulkString(slice)
        case UInt8(ascii: "*"):
            guard let line = try readLine(&buffer) else { return nil }
            guard let count = Int(line) else { throw ValkeyError.protocolViolation("array length '\(line)'") }
            if count < 0 { return .array(nil) }
            var items: [RESPValue] = []
            items.reserveCapacity(count)
            for _ in 0..<count {
                guard let item = try parseValue(&buffer) else { return nil }
                items.append(item)
            }
            return .array(items)
        default:
            throw ValkeyError.protocolViolation("unexpected marker '\(Character(UnicodeScalar(marker)))'")
        }
    }

    private static func readLine(_ buffer: inout ByteBuffer) throws -> String? {
        let readable = buffer.readableBytes
        guard readable > 0 else { return nil }
        let bytes = buffer.getBytes(at: buffer.readerIndex, length: readable) ?? []
        guard let index = bytes.firstIndex(of: UInt8(ascii: "\n")) else { return nil }
        let end = index > 0 && bytes[index - 1] == UInt8(ascii: "\r") ? index - 1 : index
        let line = String(decoding: bytes[0..<end], as: UTF8.self)
        buffer.moveReaderIndex(forwardBy: index + 1)
        return line
    }
}

public enum ValkeyError: Error, Equatable, CustomStringConvertible {
    case notConnected
    case connectionFailed(String)
    case protocolViolation(String)
    case commandFailed(String)
    case timedOut(String)

    public var description: String {
        switch self {
        case .notConnected: return "not connected to Valkey"
        case .connectionFailed(let detail): return "Valkey connection failed: \(detail)"
        case .protocolViolation(let detail): return "Valkey protocol violation: \(detail)"
        case .commandFailed(let message): return "Valkey command failed: \(message)"
        case .timedOut(let operation): return "Valkey \(operation) timed out"
        }
    }
}
