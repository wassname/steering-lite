# PCA interruption recovery audit

Scope: PCA recovery only. I did not dispatch paid work, run PCA, retry a request, or edit source. I recorded and conservatively settled the interrupted reservation at its upper cost after the authorized reconciliation attempt. The dry preflight refreshed ignored `manifest.json` and `cost-estimate.json` only.
-- PI[gpt-5.6-terra]

## Evidence and reconstruction

### Terminal stage

[`slop/verification/20260920T034603Z_canonical-post-mean-diff-terminal.log`](../verification/20260920T034603Z_canonical-post-mean-diff-terminal.log) records:

> `started_at=2026-09-20T03:46:03Z`
>
> `command=PYTHONUNBUFFERED=1 just sweep --run --backend real Qwen/Qwen3.5-4B outputs/bsbench-v2`
>
> `continuation=bare,prompting,random,mean_diff cached; random+mean_diff terminal negatives; remaining canonical methods only`

It next records PCA's Modal calibration cache miss at `03:46:09Z`, then judge-request cache misses through `03:53:29Z`. There is no later PCA aggregate-stage completion in this terminal log.

### PCA calibration: completed and cached

[`outputs/bsbench-v2/cache/calibration-candidates/91e3f8bfbfaee0d156d830dedbbf7d25f41d3fd71919e2b05c1ac7fd6bc14e4f.json`](../../outputs/bsbench-v2/cache/calibration-candidates/91e3f8bfbfaee0d156d830dedbbf7d25f41d3fd71919e2b05c1ac7fd6bc14e4f.json) is a PCA candidate result with 24 prompt outputs at coefficients `[0.1, 0.2, 0.4, 0.8, 1.6, 3.2]`.

Ledger reservation `ad345c3ed6aaa65d3c6b1d6191990edd8564b1ce7b8b323792bfcfba4e609870` is `modal-calibration-candidates-pca`, upper cost `$0.884346`. It has an `estimated_at_reservation_upper` record with a 195.225266666 second receipt, so it is conservatively accounted for rather than open.

Read-only Modal evidence shows app `ap-dL5tjZAX0qJOPjjZa7r3S4` is stopped with zero tasks. Its function call completed all 24 PCA generations; its log ends:

> `2026-09-20 11:49:42+08:00 ... generation 4/4`
>
> `2026-09-20 11:56:31+08:00 Stopping app - local client disconnected. Use modal run --detach to keep apps running even if your local client disconnects.`

This is strong evidence that Modal finished the calibration before the client was stopped. It is not an invoice, so the ledger's upper estimate remains the correct conservative charge.

### Exact interruption point

The PCA candidate set expands to 96 judge *records* (six coefficients × four prompts × AB/BA × target-aware/blind); repeated record payloads mean there are 80 distinct payload keys. The ledger has 24 PCA judge reservations:

- 23 have a settlement and a matching local response cache record.
- The last completed request is reservation `cacd62a6d3428ca46ca06bb037ae79bfb03af9d4b25f6aad907545bc9f32ecaf`, request key `53856643d15f2852455fcb05cbee5d4950da418d72799390d96301b74fed4a9f`: BSV2-002, BA, target-aware, settled for `$0.0003180897` at `03:53:25Z`.
- The only raw reservation without a later settlement or estimated receipt is `4a9d062d89b3b0f6082ab6d277ace4484406c29c91637a4bcf2aa7cb6e773d70`, request key `7d5c703c718b4020c818b27f825fb0190b540bee6d58600a83f41155286da7f6`: BSV2-002, BA, blind, comparison `7fd36757b229cb4a`, upper cost `$0.0017496`, reserved at `03:53:29.270000Z`.

There is no matching response cache file or post-interruption provider evidence for `4a9d…`. The next unstarted record is BSV2-003 AB target-aware. No PCA `candidate-judgments`, `candidate-aware`, `candidate-blind`, or `candidate-health` aggregate cache exists; those stages only persist after the complete request loop.

This proves that the local harness did not persist a response for `4a9d…`. It does **not** prove that OpenRouter did not accept or bill the request: process termination can occur after external dispatch but before the local cache/settlement write.

### Provider and process state

The most recent local OpenRouter account metadata is [`outputs/bsbench-v2/provider-evidence/openrouter-metadata-20260920T032823Z.json`](../../outputs/bsbench-v2/provider-evidence/openrouter-metadata-20260920T032823Z.json), queried at `03:28:22Z`, before the PCA work. It cannot resolve the outcome of the `03:53:29Z` call.

No local `run_bsbench_sweep`, Modal run/serve, or Python BS-bench process is active. Modal has no active app task. Thus the only remaining provider state is the unknown OpenRouter request outcome, not an active paid job.

## Current conservative budget

A paid-disabled dry preflight produced:

| Item | USD |
|---|---:|
| Existing ledger commitment | 6.3126065478 |
| External committed smoke allowance | 2.0000000000 |
| Expected remaining work | 16.0169124000 |
| One full retry reserve | 16.0169124000 |
| Unresolved-work reserve | 4.9429520000 |
| Total upper bound | 45.2893833478 |
| Headroom below $50 | 4.7106166522 |

This remains below `$50`, but it deliberately retains the Modal upper estimate and does not pretend that `4a9d…` cost `$0`.

## Interpretation

| Hypothesis | Credence | Evidence |
|---|---:|---|
| The local process was stopped after reserving/dispatched `4a9d…`, before cache and ledger settlement. | Likely (70%) | Reservation is last ledger event; no local response record; Modal log says the local client later disconnected. |
| OpenRouter billed `4a9d…`. | Chances a little less than even (40%) | No local receipt; no request-level provider record. Local absence cannot distinguish provider non-dispatch from a completed provider request whose response was lost. |
| The existing ledger protects the money but fails to force reconciliation before future reservations. | Likely (75%) | `4a9d…` counts toward committed upper cost, but lacks an explicit `unresolved` record. New reservations check for `unresolved` events, not raw reservations. |

PCA has no valid benchmark score yet. Candidate extraction is complete and reusable; the judged PCA condition is incomplete, so neither a positive nor a negative PCA conclusion is supported.

## Blocker and bounded recovery proposal

**Blocker:** reservation `4a9d…` has an unknown external outcome. It must not be settled at `$0` or blindly retried.

After parent review, the smallest recovery sequence is:

1. Append one explicit `unresolved` ledger event for `4a9d…`, preserving its `$0.0017496` upper charge and preventing accidental continuation before reconciliation.
2. Obtain a scoped, read-only OpenRouter metadata/usage query through an approved secret mechanism. Do not source or reveal `.env`. Compare its post-interruption usage with the 23 locally settled PCA requests and all other known ledger work; record the result as provider evidence. Aggregate account usage may still be insufficient to identify a single request, so leave the reservation unresolved if the comparison is ambiguous.
3. If the outcome stays ambiguous, retain the upper charge and run only the cached PCA method after explicit authorization. Candidate extraction and the 23 persisted judge responses should be reused; the first new external request would be the affected blind BSV2-002 record. Do not launch subsequent canonical methods until PCA has a complete aggregate result and the parent reviews it.
4. Run a new paid-disabled preflight immediately before any authorized retry. Its required limit is still `< $50`; current upper bound is `$45.2893833478`.

## Authorized reconciliation result

I appended an `unresolved` event for `4a9d…` with reason `local_process_interrupted_unknown_openrouter_outcome`. The canonical OpenRouter metadata command could not run: `OPENROUTER_API_KEY` is absent from the process environment, and the tool secret-access policy denied loading `.env`. No API request was sent and no key value was exposed. The evidence is [`20260920T043036Z_openrouter-metadata-env-only-check.log`](../verification/20260920T043036Z_openrouter-metadata-env-only-check.log).

Because aggregate provider evidence was unavailable and could not attribute the request, I imported [`20260920T043036Z_pca-openrouter-conservative-upper-receipt.json`](../verification/20260920T043036Z_pca-openrouter-conservative-upper-receipt.json), settling `4a9d…` for its full reservation upper bound, `$0.0017496`. The ledger contains the reserved, unresolved, settled, and receipt-imported records; [`20260920T043036Z_pca-openrouter-conservative-upper-settlement.log`](../verification/20260920T043036Z_pca-openrouter-conservative-upper-settlement.log) quotes all four.

The post-settlement paid-disabled preflight remains `$45.2893833478`, with `$4.7106166522` below `$50`; see [`20260920T043016Z_pca-post-settlement-dry-preflight.log`](../verification/20260920T043016Z_pca-post-settlement-dry-preflight.log). PCA did not run because the credential remains unavailable to this session. This credential failure occurred before any PCA request, Modal call, or new reservation.
