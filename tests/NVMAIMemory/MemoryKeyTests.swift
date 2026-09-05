import Foundation
import Testing
@testable import NVMAIMemory

/// Keys and scopes are the security boundary: both are chosen by the model
/// or by a request header, and both are turned into backend keys. Anything
/// that could escape a scope has to fail here.
@Suite struct MemoryKeyTests {
    @Test(arguments: [
        "decisions/sync",
        "architecture",
        "tasks/2026-09/refactor-sync",
        "gotchas/metal.kernel_v2",
        "a/b/c/d/e/f",
    ])
    func acceptsOrdinaryKeys(_ raw: String) throws {
        #expect(try MemoryKey(validating: raw).rawValue == raw)
    }

    @Test(arguments: [
        "",                       // nothing to store under
        "   ",                    // whitespace only
        "/absolute",              // would look rooted to a backend
        "../escape",              // traversal
        "a/../../etc/passwd",     // traversal, deeper
        "a//b",                   // empty segment
        "a/./b",                  // relative segment
        "with space",             // whitespace inside a segment
        "colon:key",              // the backend's own separator
        "new\nline",              // control character
        "star*",                  // glob, which would widen a scan
        "question?",
        "brace{a,b}",
    ])
    func rejectsUnsafeKeys(_ raw: String) {
        #expect(throws: MemoryError.self) { try MemoryKey(validating: raw) }
    }

    @Test func rejectsOverlongKeysAndDeepNesting() {
        #expect(throws: MemoryError.self) {
            try MemoryKey(validating: String(repeating: "k", count: MemoryKey.maximumLength + 1))
        }
        let deep = (0..<(MemoryKey.maximumSegments + 1)).map { "s\($0)" }.joined(separator: "/")
        #expect(throws: MemoryError.self) { try MemoryKey(validating: deep) }
    }

    @Test func trimsSurroundingWhitespaceRatherThanRejecting() throws {
        // Models emit trailing newlines in tool arguments constantly; that is
        // not an attack, it is formatting.
        #expect(try MemoryKey(validating: "  decisions/sync\n").rawValue == "decisions/sync")
    }

    @Test func categoryIsTheLeadingSegment() throws {
        #expect(try MemoryKey(validating: "decisions/sync").category == "decisions")
        #expect(try MemoryKey(validating: "architecture").category == "architecture")
    }

    @Test(arguments: ["", " ", "..", ".", "with space", "colon:name", "a/b", "star*"])
    func rejectsUnsafeScopeComponents(_ raw: String) {
        #expect(throws: MemoryError.self) {
            try MemoryScope(namespace: "nvmai", user: "local", workspace: raw)
        }
        #expect(throws: MemoryError.self) {
            try MemoryScope(namespace: raw, user: "local", workspace: "repo")
        }
        #expect(throws: MemoryError.self) {
            try MemoryScope(namespace: "nvmai", user: raw, workspace: "repo")
        }
    }

    @Test func scopeEqualityIsComponentwise() throws {
        let a = try MemoryScope(namespace: "n", user: "u", workspace: "w")
        let b = try MemoryScope(namespace: "n", user: "u", workspace: "w")
        let c = try MemoryScope(namespace: "n", user: "u", workspace: "other")
        #expect(a == b)
        #expect(a != c)
    }

    @Test func importanceAndConfidenceAreClamped() throws {
        let record = MemoryRecord(key: try MemoryKey(validating: "k"),
                                  value: "v", importance: 5, confidence: -1)
        #expect(record.importance == 1)
        #expect(record.confidence == 0)
    }

    @Test func backendFailuresAreDistinguishedFromCallerErrors() {
        // The server degrades on the first group and reports the second to
        // the model, so the split has to be explicit.
        #expect(MemoryError.backendUnavailable("x").isBackendFailure)
        #expect(MemoryError.timedOut(operation: "get", milliseconds: 100).isBackendFailure)
        #expect(MemoryError.disabled.isBackendFailure)
        #expect(!MemoryError.invalidKey("k", "why").isBackendFailure)
        #expect(!MemoryError.valueTooLarge(bytes: 2, limit: 1).isBackendFailure)
    }
}
