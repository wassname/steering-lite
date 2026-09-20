# KV-cache-gram OpenRouter timeout audit

Scope: the canonical continuation stopped at KV-cache-gram. I did not retry KV or start VJP methods. I reconciled the timed-out reservation at its full upper cost after the authorized provider check.

-- PI[gpt-5.6-terra]

| stage | expected | observed | consequence |
|---|---|---|---|
| Earlier conditions | Cache reuse | Bare, prompting, random, mean-diff, and PCA cache hits | No earlier condition was regenerated |
| KV candidate extraction | One Modal calibration candidate result | Cache result exists; reservation `a900…` has an `estimated_at_reservation_upper` receipt for 183.45274421 seconds | `$0.884346` stays conservatively committed |
| KV candidate judging | 96 judge records, each response persisted and settled | 15 records settled; request 16 timed out | Aggregate candidate judgment cache was not saved |
| Timeout evidence | Provider no-response record and explicit ledger status | `TimeoutError` after 181.61043234699173 seconds; reservation `6387…` first marked unresolved | Retry is blocked until review |
| Reconciliation | Identify provider charge or retain full upper charge | Account metadata is aggregate and cannot identify this request exactly | Settled at full `$0.0017496` upper with receipt |
| Modal tasks | No active paid task after client failure | app `ap-thdTtolDeUvqR8U3OgbIds` is stopped with zero tasks | No active Modal work |

## Evidence

The full process log is [`20260920T052144Z_canonical-continuation-run.log`](../verification/20260920T052144Z_canonical-continuation-run.log). It ends with:

> `TimeoutError: The read operation timed out`
>
> `error: recipe \`sweep\` failed on line 16 with exit code 1`

The no-response record is [`outputs/bsbench-v2/provider-evidence/94a91f91f87a3999c731952cd7d0114c54fb30894aba93d1f93cbd6cdc39bf38-TimeoutError.json`](../../outputs/bsbench-v2/provider-evidence/94a91f91f87a3999c731952cd7d0114c54fb30894aba93d1f93cbd6cdc39bf38-TimeoutError.json):

> `"elapsed_seconds": 181.61043234699173`
>
> `"payload_sha256": "94a91f91f87a3999c731952cd7d0114c54fb30894aba93d1f93cbd6cdc39bf38"`

[`20260920T053458Z_kv-cache-gram-reservation-pairs.json`](../verification/20260920T053458Z_kv-cache-gram-reservation-pairs.json) pairs every reservation created in this run by ID: 15 judge reservations settled with actual receipts, one KV Modal reservation has an upper-estimate receipt, and timed-out judge reservation `6387c6e872dedf583007129c4703208c22006b263a7cd3ce8251dd81cc07f760` now has unresolved, settled-at-upper, and receipt-imported events.

The post-timeout OpenRouter metadata query succeeded with 200 responses from key and credits endpoints. Its key usage changed from `$0.099578293` before the PCA recovery to `$0.152403026` after this timeout, while known settled charges over the interval are `$0.0527150396`. This is not an exact receipt for `6387…`: the interval also contains the earlier request `4a9d…`, which was itself conservatively charged at upper cost. The timeout receipt therefore charges `6387…` at its reservation upper `$0.0017496`, not `$0`.

## Timeout implementation and latency evidence

`audited_openrouter_request_callback` in [`scripts/run_bsbench_sweep.py`](../../scripts/run_bsbench_sweep.py) stores `next_request_at`, waits before the POST, and delegates the request. The main real-backend path wraps it in `_openrouter_read_timeout(180.0)`. The underlying adapter calls `urlopen(..., timeout=90)`, and the wrapper overwrites that with 180 seconds.

The timeout is observed at 181.61 seconds. Successful requests do not persist an elapsed-duration field, so the log cannot estimate a response latency distribution. Reservation timestamps have adjacent intervals from 3.013 to 16.002 seconds. A reservation is written before the callback waits, so these intervals do not measure POST dispatch timing. This needs a local callback-boundary timestamp measurement before another expensive continuation.

## Interpretation and smallest recovery proposal

### H1 [harness | Likely | 75%]

- **Mechanism:** An upstream response took longer than the 180-second wrapper timeout, leaving the local process without a response after the provider may have accepted the request.
- **Evidence:** The provider evidence says `TimeoutError` after 181.61043234699173 seconds; the reservation has no response cache record.
- **Contrary evidence:** The request may have failed upstream without charge; aggregate usage cannot separate this from the earlier conservatively charged request.
- **Discriminating test:** No external retry. Add elapsed request timing and a pre/post key-usage snapshot to the next authorized single-method continuation; a successful response records its exact receipt, while another timeout retains the upper charge.
- **Action:** Keep `6387…` settled at upper. Do not relaunch until review.
- **Interpretability:** yes for the timeout, no for its exact provider charge.

### H2 [measurement | Likely | 75%]

- **Mechanism:** The ledger cannot establish the intended 10-second request pace because it reserves cost before the callback waits.
- **Evidence:** Ledger reservation intervals include 3.013, 5.254, and 4.615 seconds, while the callback's `min_interval_seconds` applies after that reservation.
- **Contrary evidence:** The callback keeps `next_request_at`, which may already enforce the intended pace; no per-request dispatch timestamp was recorded.
- **Discriminating test:** Add a local `dispatch_started_at` field at the callback boundary and assert consecutive starts are at least 10 seconds apart before any external call.
- **Action:** Measure pacing locally before the next paid continuation.
- **Interpretability:** yes. It identifies the missing measurement without inferring a pace failure.

### H3 [measurement | Plausible | 40%]

- **Mechanism:** The lack of successful-response elapsed-time records prevents determining whether 180 seconds is a poor timeout or a rare upstream stall.
- **Evidence:** Only the failure record contains `elapsed_seconds`; settled receipts contain token/cost data but no latency.
- **Contrary evidence:** Fifteen new KV requests completed before the timeout.
- **Discriminating test:** Persist elapsed seconds with successful request receipts. A high tail near 180 supports a larger read timeout; a single isolated stall supports preserving the timeout but improving reconciliation.
- **Action:** Parameterize the runner's read timeout so a recovery can use 300 seconds without altering endpoint, model, or payload. The runner script is outside `source_hash()`, so this should preserve production-stage cache identities; verify that assertion before use.
- **Interpretability:** partial.

## Decision

- **Resolve condition:** met for this bounded failure recovery: all 17 created reservations have a later settlement or upper-estimate record; there is no open unresolved reservation.
- **Preflight:** paid-disabled aggregate preflight is `$46.2087488755`, leaving `$3.7912511245` to the `$50` limit.
- **Next action:** wait for review. If a retry is authorized, use KV only, reuse its completed Modal candidate cache and 15 persisted judge responses, and first implement/verify a parameterized 300-second read timeout plus local dispatch/elapsed timestamps. Do not start VJP methods before the KV result is complete and audited.
