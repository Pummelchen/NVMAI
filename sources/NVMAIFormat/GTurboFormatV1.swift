import Foundation

package enum GTurboFormatV1 {
    package static let magic = "GTURBO"
    package static let versionMajor = 1
    package static let versionMinor = 0
    package static let alignmentBytes: UInt64 = 16_384
    package static let residentHeaderBytes = 24
    package static let residentEntryBytes = 72
    package static let residentIndexMaxBytes: UInt64 = 32 * 1024 * 1024
    /// Ceiling for `packed_experts/layout.json`, which is a different artifact
    /// from the resident index and scales with experts x layers rather than
    /// with resident tensors.
    ///
    /// 128 MiB. Qwen 3.6 is ~22 MB (40 x 256 = 10,240 entries) and
    /// Qwen3.8-Flash-Next is ~54 MB (48 x 512 = 24,576), so the old shared
    /// 32 MiB bound rejected a correct install. Kept separate so raising it
    /// does not also loosen the resident index, which stays small (a few
    /// hundred KB) for every family.
    package static let packedExpertsLayoutMaxBytes: UInt64 = 128 * 1024 * 1024

    package static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert",
    ]

    package enum DType: UInt8, Sendable {
        case u32 = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
    }
}

package enum NVMAIFormatError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalid(field: String, reason: String)
    case overflow(field: String)
    case truncated(field: String)

    package var description: String {
        switch self {
        case let .invalid(field, reason): "\(field): \(reason)"
        case let .overflow(field): "\(field): arithmetic overflow"
        case let .truncated(field): "\(field): truncated"
        }
    }
}

@inline(__always)
package func gturboCheckedAdd(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else { throw NVMAIFormatError.overflow(field: field) }
    return value
}

@inline(__always)
package func gturboCheckedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw NVMAIFormatError.overflow(field: field) }
    return value
}

package enum GTurboPathValidator {
    package static func appleFilesystemKey(_ path: String) -> String {
        path.precomposedStringWithCanonicalMapping
            .lowercased(with: Locale(identifier: "en_US_POSIX"))
    }

    package static func validateRelativePath(_ path: String, field: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\0") else {
            throw NVMAIFormatError.invalid(field: field, reason: "unsafe relative path")
        }
        let components = path.components(separatedBy: "/")
        guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw NVMAIFormatError.invalid(field: field, reason: "non-canonical path")
        }
        let normalized = NSString.path(withComponents: components)
        guard normalized == path else {
            throw NVMAIFormatError.invalid(field: field, reason: "non-normalized path")
        }
    }

    package static func validateBasename(_ name: String, field: String) throws {
        try validateRelativePath(name, field: field)
        guard !name.contains("/") else {
            throw NVMAIFormatError.invalid(field: field, reason: "expected basename")
        }
    }
}