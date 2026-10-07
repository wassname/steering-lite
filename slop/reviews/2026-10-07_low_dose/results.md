# Lower-dose sampling for the main 9B report

Author PI/OpenAI. User: "many sweeps jump to zero ... same for all methods and side... or at least 1 between lowest and zero"; then "run it pls".

## Plan and expected evidence

Keep the existing model, cached vectors, questions, controls, judges, score and plot layout. Add two measured low-dose points per signed walk: first take the largest power of two no greater than that side's existing grid start, then sample one quarter and one half of it. This aligns nearby seed starts so the seed-averaged plot can retain shared points. The regular dose grid and mechanical stopping state stay unchanged.

- 21 learned methods x 3 seeds plus 20 random directions = 83 walks, four added dose/side points each = 332 evaluations.
- Expected output: 33,200 new benchmark answers and 25,200 new control answers, with all previous answers unchanged. Prompt baselines have no dose walk and stay unchanged.
- Every learned method/side should gain at least one common lower dose across its three seeds (`verify_plan.py`); most gain two. This matters because the plotted curves require common doses, not just one new point from one seed.
- Existing COMPLETE certificates lacking these points should trigger only the backfill. The original certificate is archived; vector presence is required, old rung records and boundary states are preserved, and added runtime is recorded separately and added to cumulative runtime.
- Fresh walks also collect these points. They are excluded from mechanical stop bookkeeping, not from judging or scoring.

Costs: first-dose timings plus model loads estimate 24,159 GPU-seconds, about $14.09 at $2.10/hour. Judging estimate $10-15; no invoice precision is implied. Reuse the already-benched A100-40GB/batch-200 preset.

## Predictions and limits

The additional observed points should distinguish gradual low-dose change from a real jump away from bare. A small coefficient need not give small off-axis change, especially for angular steering: its zero coefficient is a target angle, not generally the identity transform. No prediction that angular becomes admissible is made.

The sampled effect may remain below the judge's resolution, or carry an off-axis floor. Both remain valid measurements; do not smooth them away. Adding candidate doses can change best-dose scores and bootstrap intervals even though the scoring rule is unchanged.

## Verification

- `verify_plan.py`: checks coverage and shared lower-dose values against all original certificates, with prompt caches unchanged.
- `smoke_backfill.py`: runs the real tiny-model pipeline with control questions, then removes only the newly created low-dose samples to reproduce an old completed certificate. Runs the backfill and checks original vector/answer hashes, original rung dictionaries, original stopping state, exact archived certificate, added answers and cumulative timing.
- After the real jobs: verify 83 completed backfills, original certificate/rung preservation, every benchmark/control answer count, complete judge cache, new shared plotted doses, browser UAT and direct plus independent image inspection.

Smoke evidence (`smoke.log`): `BACKFILL_SMOKE_PASS real tiny model; four lower doses plus controls; original vector/answers/rungs/state unchanged; archive exact; cumulative timing preserved`. The full plan passes (`plan.log`): `PLAN_PASS new_doses=332; all learned-method sides gain >=1 shared lower dose; prompt unchanged`.

Generation finished: `walks_learned.log` contains 63 `DONE` and 63 `BRIDGE_BACKFILL_COMPLETE` records; `walks_random.log` contains 20 of each. Neither log contains a `FAILED` record. They contain 63 and 20 `cache hit vector` messages respectively: original extracted vectors were loaded rather than re-extracted. Artifact verification passed (`verify_backfill.log`): `BACKFILL_PASS walks=83 doses=332 new_answers=33200 new_controls=25200 original_files=4097 unchanged_or_exactly_archived`. Every learned method/side gained at least one shared lower dose. Original answer files and prompt certificates retain their hashes; amended walk certificates have exact archived originals, unchanged old dose records and stopping states.

Judging passed (`judge.log`): `JUDGE_COMPLETE missing=0`. Content caching reduced new requests to 23,458 main ratings and 1,576 blind ratings. Completion charges were $1.4958 + $0.0874 = $1.5832. Certificate backfill runtime totals 24,454.991 GPU-seconds, estimated $14.2654 at $2.10/hour; accounted total is about $15.85, excluding any unrecorded billing overhead. This is below the $24–29 estimate, primarily because fewer new judge requests were needed.

The report is rebuilt at http://localhost:8081/v5-9b-3seeds/index.html. `verify_report.log` records `REPORT_PASS old point metrics unchanged; 332 new points; all learned-method sides except angular have measured lower dots; angular has no common passing dose`. All 2,284 previous dose-point effect/off-axis/admissibility values are unchanged; the report now contains 2,616 points. `summary_before.json` preserves pre-addition aggregates; the full 252 MB prior report is machine-only at `.local/low-dose/points_before.json`.

## Observations and interpretation

- VJP-value +C now has C=0.25 and 0.5 below the former common minimum 1.5874; measured mean off-axis is 0.340 and 0.494. Top-k −C has C=0.0625 and 0.125 below 0.315, with off-axis 0.280 and 0.418. Values and all methods/sides: `verify_report.log`.
- Angular's lowest C=0.0078125 remains above the cutoff in every seed (off-axis 2.60–3.48). It was measured, but still has no common admissible curve. `src/steering_lite/variants/angular_steering.py::apply` returns `y - y_plane + plane_norm * target_dir`: as C approaches zero its target direction approaches b1, not the original in-plane direction. Lower-dose sampling alone does not repair that difference from an additive dose.
- Spherical now has admissible −C samples: at C=0.015625, net pushback +0.503, off-axis 0.965. The previous claim that its tested −C side never passed is superseded.
- VJP-resid and VJP-value keep their best doses and scores (−0.035 and −0.225). Several other scores improve at almost-inactive doses: mean_diff −0.438 to −0.204, cosine_gated −0.352 to −0.204. The full table is `table.md`.

Interpretation: better low-dose coverage exposes sensitivity of the score to the minimum sampled coefficient. A less negative score can come from changing almost nothing, rather than stronger selective pushback. No score rule was changed. The automatic top-five display therefore changes; top-k remains selectable.

## Plot correction and review

The old median filter moved dose markers away from their measured values and thinned curves to 16 points. For example VJP-value +C at C=0.25 was drawn at off-axis 0.494 instead of measured 0.340. Removed that filter/thinning; all retained seed-mean dots now keep their measured coordinates. Only connecting lines interpolate, including the remaining gap from bare. This presentation correction does not alter scores.

`dots_regression_before.log` failed on the old display: `AssertionError: dose dots must retain measured seed means (PI/OpenAI)`. `uat.log` passes after correction, also checking visible random fills and strokes. Direct inspection covered the exported PNG, browser screenshot and controls PNG. `visual_review.md` independently reports: "The near-bare dots and random-reference shading are visible." It notes overlapping dots, small captions, differing axis ticks between PNG/browser, and the browser introduction's overstatement "before the answers break". This is presentation review and artifact verification, not a complete scientific-validity or full-log audit. — PI/OpenAI
