# Worktree cleanup and Jev review

PI/gpt-6.1-sol, 2026-10-08. Reviewed main at `298eef0`.

## Result

- Thirteen non-main branches are pushed and their remote heads match local heads (`push-verification.log`). Pending edits were checkpointed, not revalidated or merged into main.
- Eight linked worktrees are unregistered. Their complete directories, including ignored outputs, environments and local configuration, are archived under `/workspace/2026/lite/.local/worktree-archive/2026-10-08/`. This preserves data; it does not reclaim their disk space.
- The remaining checkout is `/workspace/2026/lite/steering-lite` on current main. The main BS-bench outputs were moved into its `outputs/bsbench/` directory.
- Main already batches related questions. Its selective blind-rating pass is probably cheaper than adding blind questions to every pair. Kept `judge.py`, its request hashes and existing benchmark ratings unchanged.
- Fixed `judge_file.py` to average both answer orders with the same `judge.pair_change` used by the main benchmark. Its per-answer checks remain batched and cached.

## Push failure and repair

GitHub rejected two historical logs of 118,147,783 and 119,244,856 bytes, both introduced in the unpublished tip of `rewrite/bsbench-vjp` (`20a2f70`). No other branch contained that tip. The other branches pushed unchanged.

The unpublished tip was replaced with `e525c7c`: both logs are losslessly gzipped, with decompression checked against their original bytes. No remote history was force-pushed. Original history is retained in `/workspace/2026/lite/.local/worktree-archive/2026-10-08/rewrite-before-compression.bundle`. Sizes and commit mapping are in `log-compression.log`.

The primary checkout's pending results-metric and lockfile changes are on `feat/corda-space-steering` at `34a130a`. Its local settings, editor files and accidental `out=outputs/` directory are archived rather than published.

## What main sends to Jev

`scripts/bsbench/judge.py` sends:

- One request per answer: BullshitBench score plus five failure questions.
- One request per pair orientation: premise change plus off-axis change. Both orientations are requested.
- One request per selected blind pair: two stance questions plus the change concept. Selection follows the aware ratings.
- Legitimate control answers use their own state and rubric. Request hashing removes identical requests and reuses cached bare-answer ratings.

This already follows the central batching rule. The vendor says:

> Send every question that uses the same state in one request. You can mix question types freely.

> If the second request's questions could have been asked against the original state, ask them in the first request and let the code ignore the ones it doesn't need.

Source: TypeSafe vendor documentation, https://docs.typesafe.ai/primitives.md, fetched this session. These are vendor design recommendations, not evidence that this benchmark's long blind rubric is close to free.

## Live billing test

`probe.py` used six saved pairs from one scenario: three answer variants, two reversed pairs and an identical-answer control. It made 36 real decisions-API requests to `typesafe/jev-1.13`. Full requests, returned probabilities and usage are in `live_requests.jsonl`; totals are in `summary.json`.

To combine aware and blind questions without sharing the flaw hint, the probe moved the known flaw into the aware questions' structured instructions. It then compared isolated aware questions, standalone blind questions, and their union on the same state.

Observed input tokens across the six pairs:

- Original aware requests: 11,343.
- Aware requests with isolated flaw context: 11,865.
- Standalone blind requests: 7,755.
- Combined aware plus blind requests: 16,983.

Combining saves 13.4% against the same isolated rubric sent separately (19,620 tokens), or 11.1% against the current layout sent separately (19,098 tokens). Adding the three long blind questions costs 853 extra tokens per pair against the isolated aware request. The earlier skill's roughly 16 tokens per additional question came from shorter questions; it is not a general bound.

For speculative blind questions on every forward pair, measured break-even is a blind-selection fraction of 66.0% against the isolated layout, or 72.7% against the existing layout. This calculation assumes the six-pair token profile transfers and ignores deduplication differences. It is not a full-run cost estimate.

The saved main report has 273,200 forward dose-question pairs and at most 32,700 selected blind pairs before text deduplication, roughly 12% (`selection_counts.json`). That count includes overlap between best and strongest doses. Interpretation: speculative blind grading would probably cost more here. Keep selection unless an exact unique-request cost audit shows otherwise.

## Score changes and limits

The largest probability-component change when batching was 0.09. Repeating unchanged isolated requests produced changes up to 0.07 (`repeat_summary.json`, 12 additional API calls). Moving the flaw into question instructions changed one component by as much as 0.18. These few calls do not separate systematic packaging effects from all run-to-run variation, and they do not establish calibration or better human agreement.

The probe is a billing and packaging check, not a comparison of judge accuracy. No generative flash judge, hand-label calibration or benchmark regrading was performed. The original request layout remains in production so the existing leaderboard is not silently changed.

## Verification of the file judge change

`verify_file_judge.py` runs the production file judge on forward, reversed and identical pairs with a separate real API cache. The API filled five unique cells. The second run reused all five:

```text
JUDGE_CACHE_CHECK file required=5 cached=5 missing=0
forward: effect=+2.97, off_axis=1.28
reversed: effect=-2.97, off_axis=1.28
identical: effect=+0.00, off_axis=0.00
FILE_JUDGE_PASS real API; both orders; identical=0; 5 deduplicated cache cells; matches judge.pair_change
```

This is an excerpt of `cli_verification.log`, with the three metric rows abbreviated here. The full rows and failure checks remain in that file. It tests the production API, cache lookup and metric path, not synthetic ratings. The first verification run failed in the verification script's output parser because one column contains several `=` characters; splitting once fixed that parser. Its CLI output is preserved in `cli_initial_parser_failure.log`.

`git diff --check` passed. ML extraction/generation tests were not rerun because neither path changed.

Independent review could not run: the fresh Anthropic reviewer received HTTP 402 from OpenRouter's key limit before reading files. The exact sanitized error and run identity are in `independent_review.md`. No provider or credential substitution was made. Verification is by the parent only.

## ML-debug audit notes

- Evidence: 36 initial requests, 12 unchanged repeat requests, and five production CLI cache cells. No training configuration, learning-rate schedule, gradients or GPU stages apply.
- Predictions/SHOULD lines: no scientific score threshold was set. The billing question was whether shared-state savings outweigh extra blind-rubric tokens. Exact billing, not a performance threshold, answers it.
- Null/control: identical pairs give zero signed effect through the shared two-order formula. Reversed pairs negate effect and preserve off-axis change. These are arithmetic checks, not human-label accuracy controls.
- Baseline: original aware requests versus isolated-context and combined requests on the same saved texts. No held-out accuracy baseline exists in this probe.
- Raw sample: every full state, rubric and probability vector is saved in `live_requests.jsonl`. One scenario limits generalisation.
- Competing explanations: duplicate-state billing is real; long blind criteria can outweigh it under sparse selection; probability drift can be request packaging or ordinary variation. The repeated unchanged requests partially test the last explanation.
- Missing evidence: representative multi-scenario hand labels; enough repeats to estimate drift; exact unique-request weighted costs for a full run. None was claimed here.
- Decision: preserve the benchmark rubric and selective blind pass. Correct the external file judge's one-order calculation using the already-maintained shared function.
