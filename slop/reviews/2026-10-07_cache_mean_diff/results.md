# Qwen3.5-9B cache mean-difference evaluation

Author: PI/OpenAI. User: "there's a new kv cache steering right? run that too?"

## Setup and predictions

`cache_mean_diff` edits the final prompt token's cached values once per selected full-attention layer. It does not edit keys or each newly generated value. The first answer token remains unsteered; later tokens can attend to the altered prompt cache. Extraction uses the existing nonsense-question persona pairs, not the paper's CoT demonstrations or offset-token protocol. Reference and implementation details: `src/steering_lite/variants/cache_mean_diff.py` (Belitsky et al., https://arxiv.org/abs/2507.08799).

Run command: `just sweep cache_mean_diff`. This selects the already-benched `qwen3.5-9b` A100-40GB/batch-200 preset, full cohort, seeds 0/1/2, `--pairs bsbench_v1 --controls`. The shared walk adds two lower doses per sign. Other methods and random directions are not rerun. Code provenance: `source_commit.txt` plus `source_worktree.patch`.

Prediction: cached continuation scoring should detect nonzero KL after the first token. Prefill-only scoring would miss the intervention. A small or null behavioral effect will not by itself establish a failed implementation; compare calibration, actual outputs, zero-dose behavior and selected layers first. No prediction that this method beats the residual methods is made.

Budget expectation: roughly $10 for three walks plus judging, based on recent per-method costs, not a spending guarantee. Measure actual certificate runtime and judge charges. No new preset was introduced; inspect peak memory if allocation fails.

## Integration checks

The four handover commits were cherry-picked into the development branch, not literal main: `d78ce35`, `7cc060d`, `004dd7c`, `fb1fd14`.

- `library_smoke.log`: "68 passed in 76.65s". Real registered-method extraction, application and save/load smoke.
- `typed_cache_smoke.log`: "11 passed, 57 deselected" under `BEARTYPE=1`. Targeted cache tests include continuation behavior; this is not a passing full typed suite.
- `walk_smoke.log`: "SMOKE_PASS method=cache_mean_diff rungs=2 bridge_doses=4". Real CPU benchmark pipeline, including lower doses and controls, completed.

## Current state

All three 9B walks completed and were downloaded. `verify_walks.log` records `CACHE_WALKS_PASS seeds=3 controls=100 lower_points=12 dose_points=116 seconds=8024.498 estimated_GPU_USD=4.6810`. Each dose has all 100 benchmark and 100 control answers; all sides include C=2 and C=4 below their regular grid starts. Estimated completed GPU runtime cost is $4.68; judge charges remain pending.

Observed calibration coefficients are +82.91 to +88.17 and −99.27 to −107.52 (magnitudes stored in `walk_summary.json`); these are method-specific cache-value units, not comparable to residual-vector coefficients. The `FINAL` demo's KL is labeled "this prompt only", not the aggregate calibration target.

Read the first benchmark scenario at the lowest, middle and highest dose for every seed/sign, with controls (`answer_samples.md`, selected by position, not score). At low/middle doses all examples still accept its made-up premise; wording changes. At the largest doses several outputs loop, leak role tokens or change language. This single scenario does not determine the aggregate method result. The Transformers messages marked `[ERROR]` concern missing output-dataclass docstrings; all processes continued and completed.

The maintained `just results` command completed: `judge_report.log` records `JUDGE_COMPLETE missing=0`, 2,732 report points and `UAT_PASS`. `verify_report.log` records `CACHE_REPORT_PASS seeds=3 points=116 common_curves=2 lower_doses=2,4 controls_present; README asset matches plot`. The live report is http://localhost:8081/v5-9b-3seeds/index.html.

## Result

| selection | directed effect↑ | off-axis↓ | C |
| --- | ---: | ---: | ---: |
| best-score −C | +0.0063 | 0.1997 | 2 |
| best-score +C | −0.0113 | 0.1965 | 2 |
| largest admissible −C net pushback | +0.0741 | 0.3760 | 25.3984 |
| largest admissible +C effect | +0.0169 | 0.2411 | 4 |

Source: `metrics.json` and `verify_report.log`; admissible means all three seeds pass the unchanged mean off-axis ≤1.5 cutoff. Headline score −0.2078, 90% CI [−0.2602, −0.1652]. Legitimate rejection at the selected −C dose is 3.16%, versus 2.97% bare. The automatic top-five view includes the method.

Interpretation: this adaptation produces little directed premise change across its shared admissible doses. Its relatively high score comes from choosing the lowest, nearly inactive dose, not effective bidirectional steering. Larger doses change wording/style and eventually cause obvious generation failures. This exposes the score's preference for near-inactivity when alternatives have a larger off-axis cost; scoring was not changed for this run.

No claim that the paper is invalid, or that all bugs have been ruled out. Real-model continuation-logit assertions in `walk.py:660–670` passed, and calibration detects a nonzero intervention. They do not establish semantic sensitivity. Plausible limitations include the unsteered first answer token, editing just one prompt-cache position, and using persona contrasts rather than the paper's demonstrations; none was isolated here. This is a bounded artifact/sample inspection, not a completed full-log scientific audit.

## Costs and presentation checks

- GPU runtime estimate: $4.6810 at $2.10/hour, from completed certificates.
- Recorded judging charges: $1.1679 for 18,064 main requests and $0.0026 for 47 blind requests (`judge_report.log`).
- Accounted total: $5.8515, excluding unrecorded billing overhead.

Direct inspection covered `plot.png`, `page.png`, and `controls.png`. `visual_review.md` independently reports "cache_mean_diff stays near zero premise change while accumulating off-axis change" and confirms visible measured dots/random bands. It flags overlapping near-origin points, similar purple cache/prompt colors, and crowded labels; shapes and labels distinguish them. The browser now explicitly warns that nearly inactive doses can score well. `uat_final.log` passes after that text correction. The review does not certify scientific validity.
