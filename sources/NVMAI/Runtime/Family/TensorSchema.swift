import Foundation

/// Maps a model family's logical tensor roles to its on-disk tensor names.
///
/// One schema value per `ModelFamily`, defined in that family's file in this
/// directory. `Model`'s accessors resolve names through the schema, so
/// supporting a family with different naming (or extra roles) means adding a
/// schema — never editing shared accessors. Roles a family does not have are
/// simply absent from its schema file; asking `Model` for them fails at the
/// resident-index lookup with the exact missing name, which is the error a
/// misconfigured install should produce.
public struct TensorSchema: Sendable {
    /// Per-layer roles shared by the current families. Extend by adding
    /// closures here with a default that mirrors the Qwen3.5-MoE names, and
    /// override in the family file that differs.
    public let embedding: String
    public let lmHead: String
    public let finalNorm: String
    public let qProj: @Sendable (Int) -> String
    public let kProj: @Sendable (Int) -> String
    public let vProj: @Sendable (Int) -> String
    public let oProj: @Sendable (Int) -> String
    public let router: @Sendable (Int) -> String
    public let sharedExpertGate: @Sendable (Int) -> String
    public let sharedExpertUp: @Sendable (Int) -> String
    public let sharedExpertDown: @Sendable (Int) -> String
    public let sharedExpertScalarGate: @Sendable (Int) -> String
    public let inputNorm: @Sendable (Int) -> String
    public let postAttnNorm: @Sendable (Int) -> String

    public init(embedding: String,
                lmHead: String,
                finalNorm: String,
                qProj: @escaping @Sendable (Int) -> String,
                kProj: @escaping @Sendable (Int) -> String,
                vProj: @escaping @Sendable (Int) -> String,
                oProj: @escaping @Sendable (Int) -> String,
                router: @escaping @Sendable (Int) -> String,
                sharedExpertGate: @escaping @Sendable (Int) -> String,
                sharedExpertUp: @escaping @Sendable (Int) -> String,
                sharedExpertDown: @escaping @Sendable (Int) -> String,
                sharedExpertScalarGate: @escaping @Sendable (Int) -> String,
                inputNorm: @escaping @Sendable (Int) -> String,
                postAttnNorm: @escaping @Sendable (Int) -> String) {
        self.embedding = embedding
        self.lmHead = lmHead
        self.finalNorm = finalNorm
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.router = router
        self.sharedExpertGate = sharedExpertGate
        self.sharedExpertUp = sharedExpertUp
        self.sharedExpertDown = sharedExpertDown
        self.sharedExpertScalarGate = sharedExpertScalarGate
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
    }

    public static func schema(for family: ModelFamily) -> TensorSchema {
        switch family {
        case .qwen36: return .qwen36
        case .qwen36MTP: return .qwen36MTP
        case .qwen38flash: return .qwen38flash
        }
    }
}
