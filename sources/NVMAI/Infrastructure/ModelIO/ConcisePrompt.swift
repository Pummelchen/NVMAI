import Foundation

/// Built-in system prompts for NVMAI "concise mode": answer directly and
/// lean, without preamble, filler, or closing codas.
///
/// The prompts are derived from the Nail-Qwen3.6-35B-A3B chat template
/// (peculiar-ragdoll), which force-appends a terseness prompt to every
/// request. Measured on M3 24 GB, temp 0, 8 chat questions:
///
///   quant | baseline | tuned | strengthened
///   ------+----------+-------+-------------
///   4-bit |  3,788+  | 1,480 |     —
///   6-bit |  3,714+  | 1,680 |     —
///   8-bit |  3,635+  | 1,570 |    759
///
/// (`+` = baseline hit the 512-token cap). The standard prompt is the best
/// measured prompt for 4-bit and 6-bit; the strengthened prompt is measured
/// best for 8-bit, which is intrinsically the wordiest quantization. See the
/// wiki FAQ and Optimization Journey for the full tables.
public enum ConcisePrompt {
    /// Best measured prompt for 4-bit and 6-bit (1,480 / 1,680 tokens for 8
    /// answers vs 3,788+ / 3,714+ baseline).
    public static let standard = """
    You are Qwen3.6-35B-A3B in concise mode. Think before answering, then answer directly.
    Lead with the answer, then include only what the answer needs to be correct and usable.
    Never: open with preamble or pleasantries; restate the question; add filler transitions; hedge with niceties; repeat a point already made; or add a closing summary, follow-up offer, or 'let me know if you have questions' coda.
    Always: keep essential steps, caveats, uncertainties, and specifics — never drop correctness or a needed warning for brevity. Keep the final answer lean: use the least structure that conveys it (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why — never omit uncertainty for brevity's sake.
    If a user request is genuinely ambiguous, ask one sharp question instead of guessing.
    When the answer is complete, stop — end with the answer itself.
    """

    /// Best measured prompt for 8-bit (759 tokens vs 1,570 for the standard
    /// prompt). The 8-bit quantization is the wordiest, so the strengthened
    /// prompt bans introductions, structure-justification, and wrap-ups.
    public static let strengthened = """
    You are Qwen3.6-35B-A3B in concise mode. Answer with the answer only — lead with it, then add exactly what is needed to be correct and usable, nothing more.
    Never: open with preamble, pleasantries, or 'here is...' introductions; restate the question; add filler transitions; hedge with niceties; repeat a point; explain or justify the answer's structure; or add a closing summary, wrap-up sentence, or follow-up offer. When the answer is complete, stop — end with the answer itself.
    Always: keep essential steps, caveats, uncertainties, and specifics — brevity never drops correctness. Use the least structure that conveys the answer (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why. If the request is genuinely ambiguous, ask one sharp question instead of guessing.
    """

    /// Select the concise prompt for a routed-expert bit width.
    public static func prompt(forRoutedExpertBits bits: Int) -> String {
        bits >= 8 ? strengthened : standard
    }

    /// Select the concise prompt for a loaded model.
    public static func prompt(for model: Model) -> String {
        prompt(forRoutedExpertBits: model.routedExpertWeightBits)
    }

    /// Insert the concise system message after the last system/developer
    /// message (so our instructions come last, matching the Nail template's
    /// append-after-user-guidance behavior), or as the opening message when
    /// the request has no system guidance.
    public static func appendingSystemPrompt(
        _ prompt: String,
        to messages: [GFTokenizer.Message]
    ) -> [GFTokenizer.Message] {
        let concise = GFTokenizer.Message(role: .system, content: prompt)
        guard let index = messages.lastIndex(where: {
            $0.role == .system || $0.role == .developer
        }) else {
            return [concise] + messages
        }
        var result = messages
        result.insert(concise, at: index + 1)
        return result
    }
}
