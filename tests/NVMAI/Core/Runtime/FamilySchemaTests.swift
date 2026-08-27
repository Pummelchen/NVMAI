import Testing
@testable import NVMAI

/// The family tensor schemas are the single source of on-disk names; these
/// pins keep a schema edit from silently un-mapping an installed `.gturbo`.
@Suite struct FamilySchemaTests {
    @Test func qwen36SchemaMatchesTheRepackedNames() {
        let schema = TensorSchema.schema(for: .qwen36)
        #expect(schema.embedding == "language_model.model.embed_tokens.weight")
        #expect(schema.lmHead == "language_model.lm_head.weight")
        #expect(schema.finalNorm == "language_model.model.norm.weight")
        #expect(schema.qProj(7)
            == "language_model.model.layers.7.self_attn.q_proj.weight")
        #expect(schema.router(0)
            == "language_model.model.layers.0.mlp.gate.weight")
        #expect(schema.sharedExpertScalarGate(39)
            == "language_model.model.layers.39.mlp.shared_expert_gate.weight")
        #expect(schema.inputNorm(3)
            == "language_model.model.layers.3.input_layernorm.weight")
    }

    @Test func familiesShareTheQwenNamingWhereVerified() {
        // The MTP sidecar reuses the layer naming, and Qwen3.8-Flash-Next's
        // shared roles keep the Qwen3.5 names (verified against both
        // checkpoint indexes; docs/qwen38-flash-next-port.md). Its additional
        // roles arrive with the P1 runtime.
        for family in [ModelFamily.qwen36MTP, .qwen38flash] {
            let schema = TensorSchema.schema(for: family)
            #expect(schema.qProj(3) == TensorSchema.schema(for: .qwen36).qProj(3))
            #expect(schema.embedding == TensorSchema.schema(for: .qwen36).embedding)
        }
    }
}
