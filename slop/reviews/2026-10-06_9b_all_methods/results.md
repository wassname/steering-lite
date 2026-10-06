# Qwen3.5-9B all-methods report

Author PI/OpenAI. Continues the interrupted run on `dev/prompt-gains-random-reference`, after commit `183f0e4`. Main page: http://localhost:8081/v5-9b-3seeds/index.html.

## Status and artifact checks

| stage | expected | observed | consequence |
| --- | --- | --- | --- |
| walks | 21 learned methods x 3 seeds | 63 COMPLETE certificates, plus prompt x 3 and random x 20 | no missing requested method/seed |
| recovery | finish corda_pca seed 1 | float64 PCA SVD rerun COMPLETE | mixed extraction precision disclosed; root cause not proved |
| questions | 100 benchmark questions, controls on learned/prompt walks | 2,284 dose/seed/side points each contain 100 benchmark question ratings; non-random certificates enable controls | no benchmark-question reduction |
| judging | all requests present | `JUDGE_COMPLETE missing=0` in both judge logs | retained refreshes finished |
| scoring | nonnegative off-axis and control formula | all point means pass formula check | checks arithmetic, not judge validity |
| presentation | plot and interactive page agree | `UAT_PASS`; independent visual review PASS after label fixes | plot is ready for review |

Reproduce the artifact check from the repository root:

```sh
uv run python slop/reviews/2026-10-06_9b_all_methods/verify_final.py
```

Observed output in `verify_final.log`:

> PASS: 21 learned methods x 3 seeds; prompt x 3; random x 20; 86 COMPLETE certificates
> PASS: 2284 dose/seed/side points x 100 questions; off-axis nonnegative; control formula reproduced
> NOTE: COMPLETE means the walk finished, not that every method has an admissible score

The complete result table is `table_final.md`; certificate paths, hashes and timings are in `certificate_inventory.json`. The first benchmark question, seed 0, lowest dose on each side of every method is reproduced in full in `answer_samples.md`. Selection was fixed before reading the answers. This is a limited artifact verification, not a complete audit of all 41,315 lines of the sweep log or of every calibration stage. Exact per-run source hashes were not stored in the walk certificates; the SVD edit and local source snapshots must be considered when reconstructing provenance.

## Results and interpretation

Score is the weaker side's best directed premise change minus off-axis change. The -C effect also subtracts three times the rise in control-question rejection. At the reported selected doses, `table_final.md` contains:

| method | score↑ [90% CI] | -C net↑ | -C raw / controls rejected | +C↑ |
| --- | ---: | ---: | ---: | ---: |
| vjp_resid | -0.03 [-0.17, +0.17] | +1.00 | +1.26 / 12% | +1.88 |
| vjp_value | -0.22 [-0.34, -0.05] | +0.59 | +0.63 / 4% | +1.90 |
| *prompt* | -0.94 [-1.21, -0.68] | +0.07 | +1.44 / 48% | +0.48 |

Bare control rejection is about 3%. Interpretation: VJP-resid has the highest observed score in this setup, and less false rejection than prompting. It does not produce more raw pushback than prompting. Separate method intervals do not prove pairwise significance; my earlier "clear best" was too strong.

`corda_pca` with all three seeds scores -0.86 [-1.02, -0.72]. `angular_steering` has no admissible side; `spherical` has no admissible -C side. Their rows stay in the table. Weak or reversed scores describe these implementations and the tested dose ranges, not the viability of the underlying methods.

## Corrections to earlier chat claims

- Off-axis is change magnitude, including style and coherence changes. The 1.5 cutoff is a reporting choice, not a measured coherence boundary. A `spherical` sample above the cutoff is readable: "You can't just say \"IP\" because the risk isn't the same for a logo, a piece of code, or a whole company." High off-axis alone does not prove incoherence.
- The grey lines are first-crossing quantiles across reaching walks, not density contours. They are capped at 1.5 by construction. Their p90 is not the user's requested p95. Connecting bare to the first dose is interpolation.
- "Only four methods push back at all" was false. For example, the table gives PCA +0.16 net; score-optimal doses are not necessarily each method's largest pushback.
- The earlier SVD diagnosis was not established. Duplicate rows are a hypothesis, not proof of why float32 failed. Completion after switching to float64 establishes recovery, not equivalence of directions or a measured 1e-6 difference.
- A reversed observed response from `sspace_ablate` does not establish a sign bug. Its implementation projects out a subspace, then optionally adds a nudge; ablation is not a signed additive vector.

## Outstanding diagnostics (no new paid runs started)

1. [ ] Requested random-point-cloud density contours. Current conditional lines are accurately labelled but do not fulfill that request. A density contour does not generally pass through (0,0); anchoring all levels there would change the construction.
2. [ ] Judge target leakage. The earlier off-axis/flip-size association could change ranks. First test: fixed bare/steered pairs regraded by an independent judge with the same rubric, before changing weights. Agreement would increase confidence; disagreement would require inspecting which changes each judge counts. Requires an API budget.
3. [ ] Angular dose semantics. `angular_steering.py::apply` uses `y - y_plane + plane_norm * target_dir`: at coefficient zero this generally is not the identity. Cheapest test is a tiny-model zero-dose vs bare check through the attachment path, then calibration-history inspection; no basis yet for declaring iso-KL calibration incompatible with all rotations.
4. [ ] S-space sign/ablation semantics and spherical minimum dose. First inspect actual hook/scaling paths on a tiny model, compare positive and negative coefficients and zero to bare. No sign flip or smaller-dose sweep has been applied to these results.

These are nonexclusive explanations: judge leakage is plausible; dose semantics are a concrete concern for angular steering; control rejection is directly measured. This report does not assign numerical causal probabilities without the discriminating checks. Earlier diagnoses should not be treated as confirmed bugs.

## Costs

`verify_final.log` sums completed certificate durations for the 19 newly added methods x 3 seeds (the six mean_diff/vjp_resid walks were cached):

> Expansion completed GPU time: 216675.795 seconds x $2.10/hour = $126.3942
> Expansion logged Jev cost: $32.3653; combined accounted estimate: $158.7595

Judge totals come from the completion rows of each refresh, not the last line of the log: main $31.0638 + blind $0.6803 + recovered seed $0.6026 + blind $0.0186. The failed attempt, setup/CPU/memory charges and any overwritten retries are excluded. This is an accounted estimate, not an invoice; it already exceeds the $100-150 estimate. No further paid work was launched during this closeout.

## Deliverables and review

- `README.md`: main 9B plot/table; earlier results collapsed, their text preserved.
- `assets/bsbench_qwen3.5-9b_main.png`: final PNG, inspected directly.
- `uat_final.log`: page/PNG point parity and unfilled open quantile paths checked.
- `fresh_eyes_final.md`: initial independent visual review.
- `fresh_eyes_recheck.md`: corrected-image review, "PASS - presentation correctness" (source uses a typographic dash); remaining crowding of right endpoints disclosed.
- `slop/specs/20261006_bsbench_eval_frozen.md`: setup and limitations updated.

The run/report is available for review. Scientific validity checks and the cloud-contour request remain open. No push or main-branch merge.
