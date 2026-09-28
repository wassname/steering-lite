## Bottom line

**The OLMo failure looks real, but “OLMo’s architecture breaks VJP” is not yet established.** The strongest explanation is that this estimator captures persona-dependent sensitivity rather than a portable direction for premise rejection. Qwen’s success need not validate the proposed mechanism.

### Ranked explanations

These are subjective probabilities for the *dominant* explanation; mechanisms can overlap.

1. **Estimator/objective mismatch and poor persona-to-task transfer — 50%.**  
   `src/steering_lite/variants/vjp_delta.py` explicitly computes `"positive[layer] - negative[layer]"`: a **difference of gradients**, not the gradient of increasing the positive-persona projection on a benchmark prompt. Its sign therefore need not increase that projection. Moreover, `_valid_mask` excludes the final token (`"positions[None, :] < real_length - 1"`), while the cotangent comes from final-token activations. Extraction differentiates a sum over interior target positions, then averages interior source gradients; deployment edits all positions.

   Supporting evidence is stronger than the brief suggests: `slop/reviews/2026-09-28_vjp_check/vjp_check.md` shows **negative** persona-prompt cosine at positive dose on every model, approximately −0.35 to −0.50. Benchmark alignment then largely disappears. Stable split-half vectors establish reproducibility, not semantic relevance. Against this explanation: Qwen’s behavioral gains are genuine evidence of usefulness, although potentially through another mechanism.

2. **Architecture-dependent transport or damage budget — 25%.**  
   Reordered normalization, QK normalization, and hybrid versus full attention could change how these sensitivity contrasts affect generation. Large KL without candour gain supports “substantial but irrelevant intervention,” not merely insufficient dose. However, architecture is confounded with training and persona semantics. Also, `scripts/bsbench/walk.py::resolve_layers` restricts cache steering to `"full_attention"` layers: Qwen and OLMo receive different numbers/distributions of cache interventions. This complicates architectural interpretation. Near-zero alignment with mean_diff does **not** explain OLMo specifically, because successful Qwen vectors also exhibit it.

3. **Systematic implementation or measurement mismatch — 15%.**  
   Batch-size agreement and split-half stability cannot exclude a consistently wrong objective, hook location, token convention, or backward path. Crucially, `walk.py::vjp_check` claims `"sign should follow the sign of C"` and measures the **last token** using a newly estimated held-out cotangent. Neither sign nor that measurement is guaranteed by the class-difference, interior-token estimator. Thus this check does not establish a broken VJP implementation. No actual gradient error is demonstrated.

4. **Dose/judge/sample-selection artifact — 10%.**  
   Full breakdown-bracketing walks, working simpler methods, and the hand reading substantially weaken this explanation. Residual uncertainty remains: one OLMo seed, selected doses, and a 25-question reading using **vjp_delta-nothink**, not the default vector.

### Best single cheap check

**Run a matched-objective, central finite-difference audit on a handful of held-out persona pairs and benchmark prompts, on OLMo and 4B.** Start with vjp_delta.

Freeze the extraction cotangent; reproduce the exact interior-token objective and source-position mask. Compare the autograd directional derivative along the stored vector with  
\[
[F(h+\epsilon v)-F(h-\epsilon v)]/(2\epsilon).
\]
Use two small, numerically resolvable step sizes; retain positive, negative, and benchmark results separately.

- **Disagreement**, especially OLMo-specific: investigate gradient/intervention implementation before interpreting architecture.
- **Agreement, strong persona response, weak benchmark response:** supports domain/objective mismatch.
- **Agreement, substantial benchmark projection but no behavioral gain:** implicates the semantic readout or downstream generation rather than failed transport.

This requires no judge or full generation walk. It cannot uniquely separate architecture from training, but best tests the currently unsupported mechanistic inference.

### Is stance stratification valid?

**Yes for diagnosing headroom within a model; no as a controlled cross-model comparison.** OLMo’s negligible improvement among 85 accepting questions rules out “already rejects” as its main explanation. But the 21 accepting 27B questions are a different, potentially harder subset than OLMo’s 85.

Other traps: selecting on a noisy baseline rating induces regression to the mean; optimizing doses on the evaluated questions creates selection optimism; ordinal score differences assume comparable spacing; and candour-only gains omit the other side and off-axis costs. The table’s `153` and `63` are repeated question–seed observations, not independent questions.

Report common-question comparisons alongside model-specific strata, select doses separately, and cluster uncertainty by question. None of these adjustments should be presumed to erase the observed OLMo gap.