import Foundation

@MainActor
enum MetricFormat {
    // D26: fixed POSIX locale so decimal separators and grouping do not vary
    // with the user's region settings.
    private static let posixLocale = Locale(identifier: "en_US_POSIX")

    private static let memoryFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter
    }()

    private static let storageFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useMB, .useGB, .useTB]
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter
    }()

    static func seconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        if value < 1 {
            return String(format: "%.0f ms", locale: posixLocale, value * 1000)
        }
        return String(format: "%.2f s", locale: posixLocale, value)
    }

    static func milliseconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        return "\(value.formatted(.number.precision(.fractionLength(1)))) ms"
    }

    static func rate(_ value: Double) -> String {
        String(format: "%.1f", locale: posixLocale, value)
    }

    static func percent(_ value: Double) -> String {
        String(format: "%.1f%%", locale: posixLocale, value)
    }

    static func perToken(_ value: Double) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1))))/tok"
    }

    static func megabytesPerToken(_ value: Double) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1)))) MB/tok"
    }

    static func memory(_ bytes: UInt64?) -> String {
        guard let bytes else { return "\u{2014}" }
        return memoryFormatter.string(fromByteCount: Int64(bytes))
    }

    static func storage(_ bytes: UInt64) -> String {
        storageFormatter.string(fromByteCount: Int64(clamping: bytes))
    }

}
