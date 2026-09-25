## Review: 58491c6 (Jev as the only judge)

I read the full changed files, ran the diff, and verified behaviour against the real data: reconstructed request hashes hit the cache 400/400 on dev manifest rows, `blind_targets` produces 1280 requests all cached, best/strongest doses are 980/980 blind-attached, and `build_points`/`choose` execute cleanly over the real dev walks. I also inspected real Jev records (`premise`/`damage` scores; blind `stance_A/stance_B/concept` with `choice`+`probabilities` incl. `accepts`/`rejects`) — all match the code's assumptions. Sign conventions (`effect = steered − bare`, `directed()` −C flip) are consistent across results.py, tables, and main.jsx. No imports of the removed modules remain.

**I found no serious silent-wrong-numbers bug in the changed code.** Findings, ranked:

**1. Medium — stale DeepSeek-era `results/full/points.json` is still consumed.**
`outputs/bsbench/results/full/points.json` has `"judge": "deepseek/deepseek-v4-flash-0731"` and old-format blind records (`{"a_premise": ..., "concept": "style"}` — verified by loading it). Two consequences:
- `cost.py:56` reads this file and picks the "top 4 methods" for the larger-model plan from DeepSeek-era scores, silently mixing judges (dev was regenerated at 15:59; full was not).
- If the full site is served with the new `main.jsx`, the explorer crashes on blind rows: `q.blind.concept.choice` (`concept` is a string) and `q.blind.stance_A.choice` (`stance_A` undefined → TypeError) (main.jsx:117).
Fix: rerun `results.py --cohort full` before `cost.py`/deploy; optionally assert `site["judge"].startswith(MODEL)` in cost.py/calibration.py so staleness fails fast per AGENTS.md.

**2. Low — bootstrap never rechecks the damage cap inside resamples** (results.py:145-158, unchanged from 7844eb5 but now gatekeeping Jev-damage admissibility). `resample` recomputes `effect`/`off_axis` but keeps `admissible: True`; a resampled dose whose re-weighted mean `steered_damage` exceeds `MAX_DAMAGE` (1.5) still counts as admissible, so "selection redone in each draw" is only partial and CIs are slightly optimistic for methods sitting near the damage cap. Only bites when per-question damage is heterogeneous near the cap. Fix: recompute `steered_damage` from the chosen questions (the field is already in `q`) and re-derive `admissible` in `resample`.

**3. Low — stale scale comment.** judge.py:34: `MAX_DAMAGE = 1.5  # ... (reference: 1.5 of 5)` — the Jev damage scale is 0-4 (results.py's intro correctly says "≤ 1.5 of 4"). Comment only, but it misdocuments the admissibility rule.

**4. Negligible — cost.py blind estimate.** `mean(blind) * 400` assumes best ≠ strongest dose (2 doses × 2 sides × 100); cache-key dedup makes it 200 when they coincide, overestimating blind cost up to 2× (≈$0.02/walk, immaterial).

**Info — no action needed:** old-rubric records (122k `sycophancy` 0-6) share `jev.jsonl` with new `premise` 0-8 records; no collision is possible because the rubric text is inside the hashed request (verified: new-key lookups hit, old records are inert). DeepSeek-era `judgments.jsonl`/`blind.jsonl`/`jev_retest.jsonl` remain in `outputs/bsbench/judgments/` but nothing reads them. The removed-judge cleanup (code, justfile, UAT judge-switch) is complete.