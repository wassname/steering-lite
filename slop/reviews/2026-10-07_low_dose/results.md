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

Next command: `bash slop/reviews/2026-10-07_low_dose/run.sh`. Learned-method jobs run first, then random directions; each stage uses Modal's existing GPU concurrency limit. Pull, verify original-record preservation, judge, then rebuild the same main report.
