import Foundation

/// A filter over one task's memory.
///
/// Every field narrows; an empty query returns the whole active set. There is
/// no cross-task query by construction, because the task is the isolation
/// boundary and a query type able to cross it would make leaking between two
/// unrelated projects a one-line mistake.
public struct MemoryQuery: Sendable, Equatable {
    public var namespaces: [String]
    /// Exact keys within the selected namespaces.
    public var keys: [String]
    /// Namespace prefix, so `plot` matches `plot.act1` and `plot.act2`.
    public var namespacePrefix: String?
    /// Case-insensitive substring over key and value.
    public var text: String?
    /// An item matches when it carries any of these tags.
    public var tags: [String]
    public var statuses: [MemoryStatus]
    public var updatedAfter: Date?
    public var minimumImportance: Double?
    public var limit: Int?
    public var order: MemoryOrder

    public init(namespaces: [String] = [],
                keys: [String] = [],
                namespacePrefix: String? = nil,
                text: String? = nil,
                tags: [String] = [],
                statuses: [MemoryStatus] = [.active, .disputed],
                updatedAfter: Date? = nil,
                minimumImportance: Double? = nil,
                limit: Int? = nil,
                order: MemoryOrder = .relevance) {
        self.namespaces = namespaces
        self.keys = keys
        self.namespacePrefix = namespacePrefix
        self.text = text
        self.tags = tags
        self.statuses = statuses
        self.updatedAfter = updatedAfter
        self.minimumImportance = minimumImportance
        self.limit = limit
        self.order = order
    }

    /// Everything currently believed, most important first.
    public static var active: MemoryQuery { MemoryQuery() }

    public static func namespace(_ value: String) -> MemoryQuery {
        MemoryQuery(namespaces: [value])
    }

    public static func search(_ text: String, limit: Int? = nil) -> MemoryQuery {
        MemoryQuery(text: text, limit: limit)
    }

    /// Whether an item satisfies everything except the limit and the order.
    func matches(_ item: MemoryItem) -> Bool {
        if !statuses.isEmpty && !statuses.contains(item.status) { return false }
        if !namespaces.isEmpty && !namespaces.contains(item.namespace) { return false }
        if !keys.isEmpty && !keys.contains(item.key) { return false }
        if let prefix = namespacePrefix, !prefixMatches(item.namespace, prefix) { return false }
        if let after = updatedAfter, item.updatedAt <= after { return false }
        if let floor = minimumImportance, (item.importance ?? 0) < floor { return false }
        if !tags.isEmpty && tags.allSatisfy({ !item.tags.contains($0) }) { return false }
        if let needle = text, !needle.isEmpty {
            let haystack = "\(item.address) \(item.value)"
            if haystack.range(of: needle, options: .caseInsensitive) == nil { return false }
        }
        return true
    }

    /// `plot` matches `plot` and `plot.act1`, but not `plotting`. Segment
    /// boundaries matter: a plain `hasPrefix` would quietly pull in a
    /// neighbouring namespace whose name merely starts the same way.
    private func prefixMatches(_ namespace: String, _ prefix: String) -> Bool {
        if namespace == prefix { return true }
        return namespace.hasPrefix(prefix + ".")
    }
}

public enum MemoryOrder: String, Sendable, Equatable, CaseIterable {
    /// Importance first, then recency. The assembler's default.
    case relevance
    case recency
    case address
}

/// What a write did, so callers can log and events can be recorded without a
/// second read.
public struct MemoryWriteResult: Sendable, Equatable {
    public let item: MemoryItem
    /// Nil when the address is new.
    public let previousVersion: Int?
    public var isNew: Bool { previousVersion == nil }

    public init(item: MemoryItem, previousVersion: Int?) {
        self.item = item
        self.previousVersion = previousVersion
    }
}
