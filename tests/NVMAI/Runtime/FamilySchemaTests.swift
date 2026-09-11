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
        // shared roles keep the Qwen3.5 names.
        let mtp = TensorSchema.schema(for: .qwen36MTP)
        #expect(mtp.qProj(3) == TensorSchema.schema(for: .qwen36).qProj(3))
        #expect(mtp.embedding == TensorSchema.schema(for: .qwen36).embedding)
    }

    /// Qwen3.8-Flash-Next deliberately does NOT share the naming. It aliased
    /// qwen36 while it was a placeholder; now that the real schema is in
    /// place, asserting the alias would assert a bug. Its names are checked
    /// against the real checkpoint index in Qwen38FlashSchemaTests.
    @Test func qwen38FlashUsesItsOwnNaming() {
        let flash = TensorSchema.schema(for: .qwen38flash)
        let qwen = TensorSchema.schema(for: .qwen36)
        // Mirror-image prefixes: model.language_model vs language_model.model.
        #expect(flash.qProj(3) != qwen.qProj(3))
        #expect(flash.embedding != qwen.embedding)
        #expect(flash.lmHead != qwen.lmHead)
        #expect(flash.embedding.hasPrefix("model.language_model."))
        #expect(flash.lmHead == "lm_head.weight")
    }
}
