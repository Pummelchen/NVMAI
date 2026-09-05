import Foundation

/// The bounds a task's memory is held to.
///
/// Bounds exist because this store is RAM-primary and lives inside the same
/// process as a model runner that wants every byte for weights. An unbounded
/// memory would be a slow leak that only shows up in the middle of a long
/// task, which is the worst possible moment.
public struct MemoryLimits: Sendable, Equatable {
    /// Largest single value. A memory item is a fact, not a file; anything
    /// larger belongs in the workspace with a memory item pointing at it.
    public var maxValueBytes: Int
    /// Bytes a task's memory may hold, live items and retained versions
    /// together.
    ///
    /// This is the real bound. Counting actual bytes rather than multiplying
    /// an item count by the largest permitted value matters by orders of
    /// magnitude: at a 64 KiB value cap and 200-byte facts, a worst-case
    /// bound refuses writes at about a third of a percent of the memory it
    /// claims to allow.
    public var maxBytesPerTask: Int
    /// Active items per task. A backstop against a pathological number of
    /// tiny facts, not the primary bound. Reaching either is a modelling
    /// error rather than a capacity problem, so both throw rather than
    /// evicting: silently dropping a fact the model wrote is worse than
    /// refusing to add one.
    public var maxItemsPerTask: Int
    /// Retained history per address, oldest dropped first.
    public var maxVersionsPerAddress: Int
    public var maxNamespaceLength: Int
    public var maxKeyLength: Int

    public init(maxValueBytes: Int = 16 * 1024,
                maxBytesPerTask: Int = 192 << 20,
                maxItemsPerTask: Int = 65_536,
                maxVersionsPerAddress: Int = 32,
                maxNamespaceLength: Int = 128,
                maxKeyLength: Int = 128) {
        self.maxValueBytes = maxValueBytes
        self.maxBytesPerTask = maxBytesPerTask
        self.maxItemsPerTask = maxItemsPerTask
        self.maxVersionsPerAddress = maxVersionsPerAddress
        self.maxNamespaceLength = maxNamespaceLength
        self.maxKeyLength = maxKeyLength
    }

    public static let `default` = MemoryLimits()
}

/// Namespace and key rules.
///
/// Addresses appear in prompts, in logs, in journal files and in dependency
/// lists that are parsed back apart on a dot. Allowing whitespace, dots in
/// keys or control characters would make each of those ambiguous, so the
/// grammar is narrow and enforced at the only entry point.
public enum MemoryAddressValidator {
    /// Lowercase letters, digits, underscore and hyphen, in dot-separated
    /// segments. `plot.act1` is a namespace; `missing_brother` is a key.
    public static func validateNamespace(_ value: String,
                                         limits: MemoryLimits) throws {
        guard !value.isEmpty else {
            throw ContinuityError.invalidNamespace(value, reason: "it is empty")
        }
        guard value.count <= limits.maxNamespaceLength else {
            throw ContinuityError.invalidNamespace(
                value, reason: "it is longer than \(limits.maxNamespaceLength) characters")
        }
        let segments = value.split(separator: ".", omittingEmptySubsequences: false)
        for segment in segments {
            guard !segment.isEmpty else {
                throw ContinuityError.invalidNamespace(
                    value, reason: "it has an empty segment")
            }
            guard segment.allSatisfy(isAllowed) else {
                throw ContinuityError.invalidNamespace(
                    value,
                    reason: "segments may use a-z, 0-9, underscore and hyphen only")
            }
        }
    }

    /// Keys carry no dots, so an address always splits into exactly one
    /// namespace and one key at the last dot.
    public static func validateKey(_ value: String, limits: MemoryLimits) throws {
        guard !value.isEmpty else {
            throw ContinuityError.invalidKey(value, reason: "it is empty")
        }
        guard value.count <= limits.maxKeyLength else {
            throw ContinuityError.invalidKey(
                value, reason: "it is longer than \(limits.maxKeyLength) characters")
        }
        guard !value.contains(".") else {
            throw ContinuityError.invalidKey(
                value, reason: "keys may not contain a dot; use the namespace")
        }
        guard value.allSatisfy(isAllowed) else {
            throw ContinuityError.invalidKey(
                value, reason: "keys may use a-z, 0-9, underscore and hyphen only")
        }
    }

    private static func isAllowed(_ character: Character) -> Bool {
        character.isASCII
            && (character.isLowercase && character.isLetter
                || character.isNumber
                || character == "_"
                || character == "-")
    }
}
