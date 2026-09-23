# Repaired mean_diff: scoped preflight review (offline)

PI/gpt-6-sol · 2026-09-23. The parent authorized the preflight edit and offline verification, **not** a paid run. New result root: `outputs/bsbench-v2-mean-diff-cap384/`; historical ledger remains `outputs/bsbench-v2/costs.jsonl`. Neither callback, scientific metric, persona pair, layer set nor dose grid changed.

## Source and cache identity

- Source hash after the scope edit: `eff29d45f8bdef895a652196eb284d47a39966447ec77471a1b8876ad5c96497` (not the proposal-time `19dd29c6…`). CLI entrypoint file SHA-256: `7f82e82a205cfe0d7fa1b2a6d04d9df641e4a19e6228a847be3048beec5a056f` (includes a fail-closed `--probe --method` rejection). Modal callback file SHA-256: `1c9a251cedf9e5a004a844fea32f7cb34d86eb9cac22204b7c9885549372a296`. [Full computed identity and old candidates](../verification/20260923_mean_diff_scope_identity.json).
- Current candidate stage key: `dfe2b83feea5622efae5a24a83e76f860be988d26868446569782c0d51794444`. Original signed candidate has the same non-code inputs but historical source hash `c2251322…` and frozen 64-token vector SHA `169c8ed9…`; read-only `peek_stage` found no exact hit in either root. The separate 384-token fixed-dose diagnostic vector lacks the signed candidate-stage identity and is not injected. `vector_cached_work` reports no paid-stage hits and 12 possible candidate magnitudes.
- The historical ledger, `run-summary.json`, manifest, cost estimate, report index/points/plots and original mean-difference vector sidecar have unchanged SHA-256 in [before](../verification/20260923_mean_diff_scope_historical_before.json) and [after](../verification/20260923_mean_diff_scope_historical_after.json) snapshots. The old vector was first added to the snapshot just after the scoped dry CLI, so only its post-dry identity is independently recorded; it is outside the new root and the scoped CLI did not open it for writing.

## Scope and cost

Actual non-paying CLI invocation:

```sh
.venv/bin/python scripts/run_bsbench_sweep.py --dry-run --method mean_diff --model Qwen/Qwen3.5-4B --out outputs/bsbench-v2-mean-diff-cap384 --ledger outputs/bsbench-v2/costs.jsonl --judge-pricing slop/verification/20260922_v4-provider-endpoint-metadata.json
```

The [saved CLI result](../verification/20260923_mean_diff_scoped_cli_dry.json) and [new manifest](../../outputs/bsbench-v2-mean-diff-cap384/manifest.json) contain only `mean_diff`, zero validated paid-cache hits and exactly two unrun GPU stages: candidate extract/generate and final target/transfer/generate. Candidate upper: 12 magnitudes × 2 signs × 4 prompts = 96 answers, 384 aware judge requests. Final: 20 numbered questions plus four disjoint two-question transfer cases, each with both signs × 3 dose multipliers = 168 answers. Only the 120 numbered final cells receive 480 aware + 240 blind requests. First-attempt bound: 1,104 judge requests. [Focused offline tests](../verification/20260923_mean_diff_scope_focused_pytest.log) pass (4/4): explicit/default scope parity and invalid scope rejection; provider probes reject `--method` before dispatch, since probing always uses direct prompting; fake calls through the real adapters cover both signs, four transfer cases, 480/240 final judgments, zero-call rerun; a transient judge that exhausts the request retry policy stops before final GPU dispatch. Actual default CLI dry output remains all eight methods, on a separate `.local/` proof root.

`cost-estimate.json` reports an existing commitment of `$41.3879182307`, new single-attempt upper `$2.9779637333`, projected `$44.3658819640`. Add the held `$1` ancillary item: `$45.3658819640`. The adapter allows three attempts per request; if every possible request used all three upper-price attempts, the bound including that ancillary item is `$47.8630339640` (below `$50`). No new GPU/stage retry allowance: the prior one is consumed and this manifest reports `remaining_reserve_usd: 0`. Judge attempts are reserved immediately before each call; exhausted candidate judging cannot dispatch the final GPU stage. These are commitment bounds, not invoices.

A whole-file offline orchestration run stopped at a 180-second window after older multi-seed random cases. One [followed isolated run](../verification/20260923_mean_diff_scope_isolated_legacy_test.log) reached the actual failure in 202.92 seconds: the legacy assertion expects two random stages (one seed), but the production runner now executes five random seeds and emitted ten random stages before the two mean-difference stages. At 30 seconds, faulthandler showed the main thread waiting for concurrent judge results while worker threads were in `cache.reserve_many`, `cache.settle` and `save_json`; that is evidence of active local judge/cache work, not an import deadlock. Temporary seed-aware fake data was used to reach that assertion and then reverted, as it is outside this scoped change. The fake-plan `multiplier` was corrected so the new mean-difference test can pass the production plan check. This is **not** a full-suite pass; only the saved targeted 4-test result and both dry CLI runs are claimed. No other method's scientific code changed.

## Pre-run predictions and interpretation

Historical best `mean_diff` cohort-eligible final score was `−0.2825` at `+C`, `1.0×` (20 questions; all six original final dose cells eligible), per [`measured-points.json`](../../outputs/bsbench-v2/results/measured-points.json). This original axis is sycophantic `+C` versus abrasive `−C`. Signed intended control, not universal premise rejection, is the comparison goal. This BS-bench run does not measure accuracy on real-method control prompts.

1. Useful signed recovery: the recalibrated 384-token candidate remains healthy across a usable signed range; a final dose has a higher cohort-eligible signed score **and** raw answers move in the intended direction relative to bare. Check separately whether `+C` more often accepts the fictional premise and `−C` more often rejects the named fiction or answers candidly. A stronger `+C` score alone is not better truthfulness; real-method accuracy is unavailable here.
2. Changed target/KL with no behavioral gain: candidate boundary, fitted RMS-KL target or predicted coefficients change, but the best eligible score and sign-specific raw behavior do not improve (or worsen). This supports the observed coordinate/dose effect without a control inference. Report the new target and both signs; do not compare only at fixed `0.8`.
3. Apparent judge gain: score rises while raw answers fail the relevant sign-specific expectation (for example, `−C` still endorses invented names or silently substitutes a real procedure, or `+C` does not become more accepting). Compare saved aware/blind request-response pairs, per-question flags and old/new cohort denominators; blind descriptions are not truth labels. Keep such a result as a judge-artifact hypothesis, not a repaired axis.

## Command held for parent inspection — do not run yet

```sh
.venv/bin/python scripts/run_bsbench_sweep.py --run --backend real --method mean_diff --model Qwen/Qwen3.5-4B --out outputs/bsbench-v2-mean-diff-cap384 --ledger outputs/bsbench-v2/costs.jsonl --judge-pricing slop/verification/20260922_v4-provider-endpoint-metadata.json
```

The parent owns this launch. Recheck the source/entrypoint hashes, pricing artifact, cache hits, historical ledger and the `<$50` bound immediately before it; if any identity changes, regenerate the scoped dry manifest and review it again. A failed GPU stage or exhausted candidate judge requires a fresh parent decision, not a silent stage retry.
