### Reconstruction — observations
The same operational protocol measures bidirectional premise steering in two different checkpoints.
The 27B baseline already rejects substantially more false premises, leaving less room for further improvement.
Every score is determined by the candour (−C) side, although substantial +C movement survives on 27B.
Thus the ranking primarily measures **additional candour above each model’s baseline**, not steering strength generally.

### Is the comparison valid?
**Yes as a fixed-protocol comparison; no as an isolated test of parameter scale or general method capability.** Baselines, depth, attention-layer counts and potentially training differ. The observed VJP declines are real *reported endpoint differences*, but their interpretation remains uncertain.

Candour headroom falls from 4.04 to 1.92 premise points. Yet headroom alone cannot explain the reversal: mean_diff improves while VJP methods decline. Conversely, strong VJP +C movement argues against wholesale steering failure. Marginal confidence intervals are not confidence intervals for paired method differences or ranking changes; calculate those directly.

Walking into incoherence establishes an endpoint, not that the geometric grid found each useful optimum. Re-selecting doses in bootstrap draws addresses variability, but does not establish held-out performance after dose selection.

### Most likely explanations — inference
Subjective probabilities for the **dominant** explanation, not statistically estimated probabilities:

- **40%: baseline/task geometry interacting with method selectivity.** Remaining 27B errors may require different interventions; candour floors compress effects. Random +C also suggests accepting nonsense is comparatively easy to induce.
- **25%: layer/target placement or dose-resolution mismatch.** Architectural alignment is imperfect, particularly for cache’s restricted layers and delta’s near-output target.
- **20%: genuine checkpoint-specific representation/learning differences.** Mean_diff may simply align better with this checkpoint’s remaining correctable errors; that is not necessarily a scale law.
- **10%: construct-validity or evaluation/selection artifacts.** Praise/contempt readouts raise concern about style versus factual correction. Judge repeatability establishes consistency, not accuracy.
- **5%: sampling variation as the dominant cause.** Only three learned seeds; question heterogeneity matters. Individual ranking claims remain less certain than the apparent pattern.

### User hypotheses
**1. Layers and capacity:** Layer mismatch is plausible and directly testable. Capacity mismatch is conditional: the brief does not identify which affected methods expose meaningful capacity controls. Larger hidden width alone does not imply greater rank is needed. First compare a small layer sweep at fixed capacity against a capacity sweep at fixed layers, using validation-selected settings and held-out evaluation.

**2. Heavier training → more superposition → weaker directions:** Possible, but neither training intensity nor superposition is measured. Strong mean_diff and VJP +C effects weaken a blanket “single directions stop working” explanation. Testing another undertrained model introduces another confound. A matched training-checkpoint comparison would be more diagnostic; a rank sweep alone cannot establish superposition.

### Cheapest discriminating checks
1. **Existing outputs:** Plot per-question, per-side dose curves; stratify by bare rejection and baseline agreement across models. Compute paired method and model-by-method contrasts. Stratification diagnoses composition, not causal scale effects.
2. **Evaluation audit:** Inspect admissible seed counts and pooling—especially random’s “3 of 8”—and whether missing/inadmissible seeds change the estimand. Blind-label a small stratified answer sample for factual correction, tone and damage.
3. **Small reruns:** Densify doses near useful peaks; verify signs, layer indexing and actual perturbations. Then run the layer/capacity ablations above.
4. **Replication:** Lock choices before held-out questions; eventually add matched-family checkpoints.