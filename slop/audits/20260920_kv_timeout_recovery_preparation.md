# KV timeout recovery preparation

Scope: no paid work was dispatched. This records the local change requested after the KV-cache-gram read timeout.

-- PI[gpt-5.6-terra]

## Change

`scripts/run_bsbench_sweep.py` now accepts `--openrouter-read-timeout`, defaulting to the existing 180 seconds. A reviewed KV-only retry can use 300 seconds without changing its model, endpoint, or request payload:

```sh
just sweep "--run --backend real --method kv_cache_gram --openrouter-read-timeout 300"
```

The callback writes one sanitized timing record for each uncached request. It contains the payload hash, model, response schema, callback attempt, dispatch start, response finish, elapsed duration, enforced wait, 10-second minimum interval, and success/failure outcome. It does not contain the API key or payload content. Failure records retain the existing HTTP/no-response details.

The cache source hash stayed `d2a8eb38eebf8b090ae0eb66c73bb9b766b3e3762860e440089529385633ee88`; the changed file is outside `src/steering_lite`, which `source_hash()` hashes. Existing production-stage cache identities therefore remain valid. The run-summary entrypoint hash will change, as intended, because the script changed.

## Local evidence

- [`20260920T054000Z_kv-recovery-focused-tests.log`](../verification/20260920T054000Z_kv-recovery-focused-tests.log): 14 focused production tests passed. The new callback test records two callback starts 10 seconds apart with a simulated 0.25-second first response and a 9.75-second enforced wait before the second start.
- [`20260920T054200Z_kv-recovery-smoke.log`](../verification/20260920T054200Z_kv-recovery-smoke.log): `just smoke` passed, 59 tests.
- [`20260920T054300Z_kv-recovery-full-offline-tests.log`](../verification/20260920T054300Z_kv-recovery-full-offline-tests.log): `just test` passed, 138 tests.
- [`20260920T054100Z_kv-recovery-cache-source-identity.json`](../verification/20260920T054100Z_kv-recovery-cache-source-identity.json): working and `HEAD` `source_hash()` values match and `src/steering_lite` has no local diff.
- [`20260920T054400Z_kv-recovery-postchange-dry-preflight.log`](../verification/20260920T054400Z_kv-recovery-postchange-dry-preflight.log): paid execution is false; total conservative upper is `$46.20874887549999`, below the `$50` limit.

## Status

The open-unresolved set remains empty from the prior KV reconciliation. No KV retry or VJP method was dispatched. The 300-second KV command above remains subject to explicit review.
