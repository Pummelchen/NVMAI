import Foundation

/// Qwen3.8-Flash-Next family knowledge. The shared roles keep the Qwen3.5
/// names (verified against both checkpoint indexes in
/// docs/qwen38-flash-next-port.md); the family's additional roles — the QSA
/// indexer, per-head attention norms, hyper-connection mixers, and the PLE
/// n-gram machinery — are added here when the P1 runtime lands, so nothing
/// about this family ever touches the shared schema or accessors.
extension TensorSchema {
    static let qwen38flash = TensorSchema.qwen36
}
