# Variant audit

## Goal
Audit every implemented steering variant against its paper/code reference and this repo's implementation. The output should distinguish real correctness from smoke-test-only success.

## Scope
In: implemented variants and extraction pools in steering-lite: mean_diff, caa, act_add, mean_centred, pca, topk_clusters, cosine_gated, sspace, spherical, directional_ablation, chars, linear_act, angular_steering, attn pools.
Out: deferred methods not implemented; new benchmark sweeps.

## Requirements
- R1: Each variant gets a separate audit task. Done means: a subagent verdict says PASS/WARN/FAIL with paper/code evidence and implementation mismatch notes. VERIFY: a stranger can see for each variant whether paper URL, code URL, and implementation match. Sneaky fail: subagent only reads smoke tests; catch by requiring paper/code comparison.
- R2: Paper/code references are checked. Done means: missing/wrong/weak references are flagged as WARN/FAIL, not handwaved. VERIFY: catalog rows get corrections or explicit "no paper/code" rationale.
- R3: Implementation is checked against math. Done means: comments/docstrings that overclaim are flagged; defensive complexity is flagged. VERIFY: each finding names file path and exact issue.
- R4: Fix catalog/code issues found by audit if small and clear. Done means: tests still pass. VERIFY: `uv run pytest tests/ -x -q` returns pass count.

## Tasks
- [x] T1 (R1-R3): Subagent audit mean_diff/CAA/ActAdd/mean_centred.
  - verify: subagent compares Panickssery/Turner/Jorgensen refs and mean_diff.py.
  - likely_fail: aliases overclaim paper equivalence; report says WARN.
  - sneaky_fail: mean-centering is algebraically identical under paired pos/neg; report must mention this.
- [x] T2 (R1-R3): Subagent audit pca/RepE/LAT.
- [x] T3 (R1-R3): Subagent audit topk_clusters.
- [x] T4 (R1-R3): Subagent audit cosine_gated/CAST-soft.
- [x] T5 (R1-R3): Subagent audit sspace.
- [x] T6 (R1-R3): Subagent audit spherical.
- [x] T7 (R1-R3): Subagent audit directional_ablation.
- [x] T8 (R1-R3): Subagent audit chars.
- [x] T9 (R1-R3): Subagent audit linear_act.
- [x] T10 (R1-R3): Subagent audit angular_steering.
- [x] T11 (R1-R3): Subagent audit extraction pools.
- [x] T12 (R4): Apply small fixes and verify.

## Context
House rules: fail-fast research code, no defensive fallbacks, no backward compat. One file per method under src/steering_lite/variants. Current tests pass 27/27 but this is not enough: audit must catch math/reference mismatches.

## Log
- 2026-04-28: Existing local paper library is docs/papers/steering; CHaRS has local files named 2026-concept-heterogeneity-aware-representation*.md.
- 2026-04-28: Audits found major overclaims: cosine_gated != IBM CAST; sspace != weight-SVD ssteer; old linear_act was Gaussian/Bures OT not Rodriguez Linear-AcT; angular URL was wrong and implementation was span(h,v), not fixed-plane Angular Steering; mean_centred was algebraically no-op.
- 2026-04-28: Fixes applied: linear_act rewritten to coordinate-wise affine OT; angular_steering rewritten to fixed-plane rotation; CHaRS now uses cluster-size marginals; last-token extraction gathers last non-pad token; attn_kq/hdiff scaled to avoid generic extractor factor-of-2; catalog/docs softened overclaims.
- 2026-04-28: Verify command `uv run pytest tests/ -x -q` returned `27 passed in 29.43s`.

## Audit verdicts

| Variant / group | Subagent verdict before fixes | Main issue found | Resolution |
|-----------------|-------------------------------|------------------|------------|
| mean_diff / CAA / ActAdd / mean_centred | WARN / mean_centred FAIL | aliases overclaimed exactness; mean_centred cancelled exactly | docs softened; mean_centred changed to `pos_mean - corpus_mean` |
| pca | WARN | RepE/LAT equivalence too strong | documented as RepE/LAT-inspired, vgel `pca_diff`-like |
| topk_clusters | WARN | folk status okay but closest priors hidden | catalog/docstring say internal cosine-kmeans baseline |
| cosine_gated | FAIL/WARN | not IBM CAST | renamed in docs as CAST-inspired soft self-gate |
| sspace | FAIL | cited weight-SVD ssteer but code was activation-diff SVD | reclassified as internal activation-subspace baseline |
| spherical | WARN | ungated core only; missing paper URL; coeff docs wrong | added paper URL and ungated-core wording |
| directional_ablation | FAIL/WARN | default was not pure ablation; not full Arditi pipeline | default coeff set to 0.0; docs say Arditi-inspired projection-out |
| chars | WARN | uniform OT marginals | added cluster-size marginals |
| linear_act | FAIL | Gaussian/Bures OT, not Rodriguez Linear-AcT | rewritten to coordinate-wise affine OT |
| angular_steering | FAIL | wrong arXiv ID; span(h,v), not fixed plane | corrected URL; implemented fixed-plane target-angle map |
| extraction pools | WARN | right-padding bug; attn_kq/hdiff factor-of-2 | gather last non-pad; scale rows by 0.5 |
| fresh-eyes review | WARN | docs/default mismatch; traceability gap | fixed `train_attn` doc; added this verdict table |

## TODO
- If audits find major mathematical mismatch, keep tests green but mark method WARN/FAIL in catalog rather than silently bending implementation.

## Errors
| Task | Error | Resolution |
|------|-------|------------|
| T1 | `mean_centred` subtracted corpus mean from both pos and neg, cancelling exactly. | Changed to `pos_mean - corpus_mean` using pos∪neg corpus; catalog documents equal-group half-diff degeneracy. |
| T4 | `cosine_gated` was labeled CAST-soft / IBM CAST. | Reworded as CAST-inspired soft self-gate, not full CAST. |
| T5 | `sspace` cited weight-SVD ssteer source but implementation is activation-diff SVD. | Reclassified as internal activation-subspace baseline. |
| T8 | CHaRS used uniform OT marginals. | Added kmeans cluster-size marginals. |
| T9 | `linear_act` implemented Gaussian/Bures OT, not Rodriguez Linear-AcT. | Rewrote to coordinate-wise sorted affine OT (`omega`, `beta`). |
| T10 | Angular citation pointed to optics paper and implementation used span(h,v). | Corrected arXiv ID to 2510.26243 and implemented fixed-plane rotation. |
| T11 | `record_activations` used padded `h[:, -1, :]`; attn_kq/hdiff doubled under generic pos-neg extraction. | Gather last non-pad token; scale attn_kq/hdiff rows by 0.5. |
