import Foundation

/// A safetensors file, memory-mapped.
///
/// NVMAI serves its own models from the GTurbo format, which is built for
/// streaming experts off SSD. The side-engine's model is a different
/// problem: two gigabytes that stay resident, produced by this project's own
/// converter, and read start to finish for every token. Mapping the
/// safetensors file the converter already writes is the whole loader, and it
/// avoids a second conversion step whose only job would be to restate the
/// same bytes.
///
/// Mapping rather than reading matters here. The file is larger than the
/// side-engine's whole memory budget would be if it were copied, and the
/// kernel's page cache is exactly the right owner of pages that are read
/// sequentially and never written.
public struct SafeTensorsFile: Sendable {

    public struct Entry: Sendable, Equatable {
        public let dtype: String
        public let shape: [Int]
        /// Offsets relative to the start of the payload, not the file.
        public let start: Int
        public let end: Int

        public var count: Int { shape.reduce(1, *) }
    }

    public enum Failure: Error, CustomStringConvertible {
        case unreadable(String)
        case malformed(String)
        case missing(String)
        case unsupported(dtype: String, name: String)

        public var description: String {
            switch self {
            case .unreadable(let path): "cannot open \(path)"
            case .malformed(let detail): "malformed safetensors: \(detail)"
            case .missing(let name): "no tensor named \(name)"
            case .unsupported(let dtype, let name): "unsupported dtype \(dtype) for \(name)"
            }
        }
    }

    public let url: URL
    public let entries: [String: Entry]
    private let mapping: Mapping
    private let payload: Int

    /// The mapping's lifetime, so the pointer stays valid for as long as any
    /// tensor view taken from it.
    ///
    /// unchecked-invariant: `base` and `length` are `let`, set once before
    /// the instance escapes, and describe a read-only `MAP_PRIVATE` mapping
    /// that nothing ever writes to. The only mutation in the type's life is
    /// `munmap` in `deinit`, which by definition runs after the last
    /// reference is gone.
    private final class Mapping: @unchecked Sendable {
        let base: UnsafeRawPointer
        let length: Int
        init(base: UnsafeRawPointer, length: Int) {
            self.base = base
            self.length = length
        }
        deinit { munmap(UnsafeMutableRawPointer(mutating: base), length) }
    }

    public init(url: URL) throws {
        self.url = url
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw Failure.unreadable(url.path) }
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 8 else {
            throw Failure.unreadable(url.path)
        }
        let length = Int(status.st_size)
        guard let raw = mmap(nil, length, PROT_READ, MAP_PRIVATE, descriptor, 0),
              raw != MAP_FAILED else {
            throw Failure.unreadable(url.path)
        }
        // Sequential, and read exactly once per token: tell the kernel so it
        // reads ahead rather than faulting a page at a time.
        madvise(raw, length, MADV_SEQUENTIAL)
        let base = UnsafeRawPointer(raw)
        mapping = Mapping(base: base, length: length)

        let headerLength = Int(base.loadUnaligned(as: UInt64.self))
        guard headerLength > 0, 8 + headerLength <= length else {
            throw Failure.malformed("header length \(headerLength) does not fit \(length)")
        }
        payload = 8 + headerLength
        let json = Data(bytes: base.advanced(by: 8), count: headerLength)
        guard let object = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Failure.malformed("header is not an object")
        }
        var parsed: [String: Entry] = [:]
        parsed.reserveCapacity(object.count)
        for (name, value) in object where name != "__metadata__" {
            guard let fields = value as? [String: Any],
                  let dtype = fields["dtype"] as? String,
                  let shape = fields["shape"] as? [Int],
                  let offsets = fields["data_offsets"] as? [Int], offsets.count == 2 else {
                throw Failure.malformed("entry \(name)")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0],
                  payload + offsets[1] <= length else {
                throw Failure.malformed("entry \(name) runs past the file")
            }
            parsed[name] = Entry(dtype: dtype, shape: shape,
                                 start: offsets[0], end: offsets[1])
        }
        entries = parsed
    }

    public func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else { throw Failure.missing(name) }
        return entry
    }

    /// A raw view of one tensor. Valid for the life of this file.
    public func bytes(_ name: String) throws -> UnsafeRawBufferPointer {
        let entry = try entry(name)
        return UnsafeRawBufferPointer(start: mapping.base.advanced(by: payload + entry.start),
                                      count: entry.end - entry.start)
    }

    /// A tensor as `Float`, whatever it is stored as.
    ///
    /// The small tensors -- every norm, `A_log`, `dt_bias`, the convolution
    /// taps -- are F32 or BF16 and are read once at load, so a copy is the
    /// right shape here. The large ones are quantized and are never read
    /// through this path.
    public func floats(_ name: String) throws -> [Float] {
        let entry = try entry(name)
        let raw = try bytes(name)
        switch entry.dtype {
        case "F32":
            return Array(raw.bindMemory(to: Float.self))
        case "BF16":
            // Widened by bit pattern: the top sixteen bits of a float32 are
            // exactly a bfloat16, which is the whole point of the format.
            return raw.bindMemory(to: UInt16.self).map {
                Float(bitPattern: UInt32($0) << 16)
            }
        case "F16":
            return raw.bindMemory(to: Float16.self).map(Float.init)
        default:
            throw Failure.unsupported(dtype: entry.dtype, name: name)
        }
    }
}
