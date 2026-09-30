# Follow-up evidence review: prompt embedding dev experiment (revised results.md)

Reviewer: reviewer-anthropic, read-only. Scope: revised results.md, selected-controls.json, coverage.log, prompt-cross-run-drift-judge.log, RESEARCH_JOURNAL.md 2026-09-30 entry.

## B1 — resolved

Revised results.md "Selected doses versus controls" table and `selected-controls.json` now report, per selected point, the same-persona gain-0 and opposite-persona same-gain effects (short −C gain 8: −0.6135 vs −0.4170 vs −0.6110). The text explicitly states this "does not establish an abrasive-instruction effect" and that the benchmark score "includes an apparently persona-insensitive negative shift". The journal entry repeats the same denial. That is the reporting change I asked for.

I accept the parent's correction on scope: my earlier wording ("content-agnostic perturbation plus noise" as *the* cause; "instruction content contributes essentially nothing") was broader than the evidence. Short +C gain 1 (+3.57 vs −0.26 opposite) and engineered −C gain 4 (−1.74 vs +3.71 opposite) clearly separate personas. The revised text correctly lists prefix perturbation, regeneration drift and judge sensitivity as *competing* explanations and names the missing neutral-prefix control. No remaining overclaim on this point.

## B2 — resolved

Revised text reports the blind stance shift (+0.1745), the mean `rejects_premise` label probability (0.11, explicitly "not a rejection rate") and top labels, and characterises this as "weak corroboration, not a logical contradiction" of the aware +0.6135. I agree: I used "contradicts" loosely; the two metrics are on different scales and the blind result is weak support rather than opposition. The disclosure is now adequate for a reader to weigh it.

## Minor items

- Cost: `prompt-cross-run-drift-judge.log` shows `cost=$0.0008`; 0.0287 + 0.0008 = 0.0295. Reconciled.
- Uniqueness: `coverage.log` reads `COVERAGE_PASS: 720 unique rows; complete 20-question x 9-gain x 2-sign x 2-method grid; fresh engineered identity 40/40; one engineered process`; verification.json has `unique_rows: 720`. Resolved (verify_coverage.py itself not re-read).
- Marker: `selected-controls.json` records `short_negative_gain4` damage 1.4115, admissible true, both markers circles; my visual "×" reading was wrong and the frontend selector fix is documented. Withdrawn.
- The drift-judge log additionally shows prior-process scaled_C1 vs historical premise mean_abs 0.024 while fresh vs historical is 0.243 — consistent with drift being cross-process, not scaling-induced. Not cited in results.md; optional.

## Residual (unchanged, correctly disclosed)

Single seed/process; same-data selection; CI excludes cross-process variance; no neutral-prefix control; 0–0.125 gap untested. None of these are blockers for a bounded dev note.

**Verdict:** B1 and B2 resolved; no blockers remain.