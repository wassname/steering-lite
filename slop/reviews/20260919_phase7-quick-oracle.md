# Quick Oracle Review — Commit 3dc459e (Phase-7 offline graph slice)

## Observations (evidence-checked)

**Sycophancy persona source — exact match.** Reference `scripts/walk.py` (`/workspace/2026/jspace/j-steer_pub`) declares `PERSONAS = [("sycophantic", "abrasive")]`, `PERSONA_TEMPLATE = "Answer as someone who is {persona}."`. `sweep.py` and `generation.py` both replicate this verbatim (`BSBENCH_PERSONAS = (("sycophantic","abrasive"),)`, same template, seed 0, n_pairs 256, thinking True), and `pipeline.py`'s prompting control passes `persona="sycophantic"`. Source provenance is sound and content-hashed into stage identity.

**Phase-A routes/item counts — match spec.** `dry_manifest`: bare/prompting → 4 stages (generation-20, health-20, target-aware-20, blind-20, persona_source None); six vector methods → 4 stages each (calibration-candidates-4, candidate-health-4, candidate-aware-4, candidate-blind-4) over `BSV2-001..004` (via `CALIBRATION_CASE`). 8 methods × 1 GPU stage = 8 GPU stages; tests assert exact counts and the 20-guestion numbering (`[1..20]`).

**Cache invalidation scope — purpose-built.** `cached_stage` keys on stage/model/data/method/config/prompts + `code_sha256`; `test_stage_cache_invalidates...` confirms each dimension independently invalidates. Phase-4's `cached()` is left intact (the plan's cache-semantics argument). Final-stage identity is order-stable (sorted coefficients, content-keyed case hashes); `test_final_stages_carry_complete_data_flow` confirms reordering invariance.

**Fail-fast placeholder handling — passes.** `final_stages` raises on missing/empty vector, observed, prompt_spec, case→prompts, and on `-placeholder` transfer datasets *before* any dispatch; tests cover each branch.

**Budget accounting — accurate for what it counts.** `target_aware=128` (2/pair bite: 2×20×2 for bare/prompting + 2×4×6 for vector), `blind=128`, `persona_validation=12`, `token`, 8 GPU stages; upper bound < $50; `reserve_budget` rejects at ≥$50 (strict). Tests assert counts and the no-GPU/no-blind deltas.

**Tests green.** `20260919_phase7-sweep-focused.log` 6 passed; `...-full.log` 84 passed.

**No doc overclaim.** `dry_manifest` is `mode: dry-run`, `paid_execution_enabled: False`; spec names the full sweep as goal-4 remaining. Nothing claims live dispatch is implemented.

## Inference-separated concerns

1. **Final-stage GPU cost absent from the dry-manifest budget.** `cost_estimate` is computed only over the phase-A stage list. Each vector method's `final-generation` GPU run (6 extra GPU stages) is not included, so the manifest's $<50 estimate undercounts the real paid sweep (actual ~14 GPU stages + final judging). Preflight will only catch it when final stages are reserved, not at plan time. Low severity for a non-dispatch slice; flag for senior review before `paid_execution_enabled` flips.

2. `code_sha256 = source_hash()` is a coarse global hash of `src/steering_lite/**/*.py` — any unrelated module edit busts the entire sweep cache. Conservative/safe but costly; acceptable as "safe by default."

3. `four-item candidate batch (BSV2-001..004)` models generation item counts; the extraction pair volume and calibration coefficient sweep aren't line-itemed in GPU-hours. Priced under a flat per-stage hour constant, so fine for planning.

4. **Placeholder transfer cases** (`OTHER-A`, `OTHER-B` `*-placeholder`) remain unresolved in this slice — consistent with a bounded offline phase-A/B slice, not full goal-4 completion. No overclaim, but they must be resolved before live dispatch.

## Missing evidence / unfinished

- No log ran `final_stages` end-to-end or `fit_target`/`useful_coherent_boundary` with real observed records; only manifest/identity/rejection branches are exercised (6-test focused set). The actual RMS-KL fit and coefficient prediction code path is untested in this slice.
- The prior final-answer request aborted mid-draft; no senior SSR-review integration beyond the local green suite was re-confirmed after abort.

## Verdict

**accept.** The slice is internally consistent, matches the reference persona source and the phase-7 spec, carries correct cache-identity and fail-fast semantics, is honestly scoped to dry-run planning, and tests are green. No blocking defect.

**Smallest corrections (non-blocking, do before enabling paid execution):**
1. Extend the dry-manifest budget to include the 6 per-method `final-generation` GPU stages (+ final judging requests) so the $<50 estimate reflects the real paid sweep.
2. Add one test exercising `fit_target`/`useful_coherent_boundary`/coefficient prediction with synthetic observed records (currently untested).
3. Resolve or gate the two `-placeholder` transfer cases explicitly in the live manifest.