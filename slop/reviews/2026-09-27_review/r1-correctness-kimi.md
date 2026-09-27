All evidence gathered. Verified by reading the full `results.py` score path and the cache format.

## Review: bsbench-v3 diff (1d2ba7a..HEAD + working tree)

**Room score vs existing score machinery — checked, consistent.** `room_score(best)` (results.py:141) divides at the same Pareto-best points returned by the unchanged `pareto_score`; `resample` (results.py:162) recomputes `room` over the same drawn `chosen` pooled means used for `effect`/`off_axis`; `bootstrap` (results.py:173-180) takes the room CI from the same draws, and `room_score` consumes no `rng`, so the existing `scores` sequence and CI indices (`int(0.05*len)`, `int(0.95*len)-1`) are bit-identical to before. The only changes on the score path are additive keys (`"room"`, `"bare_premise"`). Existing score is unchanged.

**Room=0 division crash — checked, not reachable.** `room_score` divides by `point["room"]`; Python raises `ZeroDivisionError` on `/0.0`. But `room` is a pooled mean over all cohort questions (`method_curve` results.py:122-123); data shows `bare_premise` min 0 but cohort means 1.9-4.0 (`outputs/bsbench/results/*/points.json`), so a zero mean is unreachable. Residual risk only.

**CONCEPTS/`probabilities` contract — checked, holds.** Cached Jev choice answers do carry `probabilities` keyed by criteria name (verified in `outputs/bsbench/judgments/jev.jsonl`), and `key()` hashes the full request so the CONCEPTS/DAMAGE-v2 edits are deliberate cache misses requiring `judge.py --refresh` — matching the `assert ... run judge.py --refresh` design in `build_points`/`blind_summary`.

**Findings:**

- **P2 — stale caption contradicts new INTENDED labels.** results.py:597 still writes `"Intended label: +C sycophantic, -C candid."` into `index.md`, but `INTENDED = {"+C": "accepts_premise", "-C": "rejects_premise"}` (results.py:506) and the blind-table column is now "P(intended label)" of mean probabilities, not top-label share. Every regenerated `index.md` will misdescribe what its own blind table reports.

- **P2 — UAT label-distribution assertion can false-fail.** web/uat.py:42 `...first.inner_text().count("%") >= 3` assumes the first blind row shows ≥3 percentages (P(intended) + ≥2 labels ≥2%). A near-deterministic row (e.g. one label at 100%, all others <2%) renders only 2 "%" and fails despite correct page behavior. Low probability with 16 labels, but it's a data-dependent test on real output.

No P0/P1 issues found in judge.py v2 rubric (DAMAGE v2 correctly moves vagueness to level 3 per CONFOUNDS "Rate vagueness as severe damage"), `--model`/`default_model_dir` glob (`Qwen/Qwen3.5-27B` → `Qwen--Qwen3.5-27B-g*`, matches on-disk dirs), blind_summary mean-probability semantics, or the main.jsx Blind table (column counts balanced: 14 headers / 14 cells).

Merge verdict: OK with notes