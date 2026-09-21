# KV final-judge second-timeout and cache audit

Target: `kv_cache_gram` final judgment recovery, run from `rewrite/bsbench-vjp` at `22be80ddb621ac164a74a12239de195cc99f4f71`.

- worker: PI[gpt-5.6-terra]
- primary run: `slop/verification/20260921T130100Z_kv-final-judge-second-retry.log` (400 lines, read in full)
- final generation: `outputs/bsbench-v2/cache/final-generation/f05418b39b6a1052395312d199661ac4c28c0231b18720511c2e0c6f52164191.json`
- code path: `scripts/run_bsbench_sweep.py` → `production._final_judgments` → `LocalJudgeAdapter._complete_one`

## Stage table

| stage | expected | observed | expected? | clues | missing metric | consequence |
|---|---|---|---|---|---|---|
| KV candidate and generation | cache reuse; no new GPU | candidate stages reuse; `f05418…` has 84 answers and health records | yes | retry log lines 4–52; final-generation record | Modal invoice for prior generation remains separate | no new GPU work in this retry |
| prior final judgments | reuse the first interrupted retry's completed requests | 178 judge-request cache hits | yes | retry log lines 42–219 | request-count summary was absent before dispatch | the first 178 exact request identities are stable across processes |
| `cde040…` recovery | retry original 179th request | provider evidence records success at retry attempt 1, 2.889 s | yes | `cde040…-000001-success.json` | none | original timeout did not repeat |
| remaining final judgments | no work only if completed already; otherwise sequentially finish remaining rows | 114 further success requests, then timeout on retry attempt 115 | partly | retry log lines 220–334; provider evidence chronology | top-level final-judgments cache is absent until all requests finish | the "cde-only" premise was false: work continued from global request 180 |
| final judgment #293 | strict schema response before 300 s | `4e8bc1…` waited 300.565 s and raised `TimeoutError` | no | `4e8bc1…-000115-failed.json`; log lines 335–400 | provider-side bill/result is not attributable | do not dispatch another judge or VJP request |
| ledger reconciliation | account for unknown provider outcome without changing pricing | exact stored upper `$0.0022643999999999997` settled; active unresolved and overage are empty | yes | `costs.jsonl` tail; receipt and failed-import log | provider invoice | conservative accounting is restored |
| cache semantics | identical payload should not trigger duplicate provider calls when a cached response exists | payload `4e8bc1…` has one request key `bbc632…`, but 16 cache records distinguished by different `comparison_id`s | no | cached judge records enumerated in verification shell output | total unique-payload vs per-row request-count preflight | paid work is duplicated for semantically identical messages |

## Chronological evidence

The final generation is complete, but final judgments are not a 84-item stage: each planned row creates AB, BA, aware and blind requests.

> `plan 84 answers 84`
>
> `for row in rows:`
>
> `    for order in ("AB", "BA"):`
>
> `        for blind in (False, True):`

The first quote is direct inspection of the persisted final-generation record. The code quote is `src/steering_lite/benchmark/validation.py:43-54`. Together they establish 336 planned request records, not 179.

The second retry begins with preserved cache records and then crosses into unfinished work:

> `2026-09-21 21:02:52.668 | INFO ... cache hit judge-request cf9993983ba3`
>
> `2026-09-21 21:02:52.669 | INFO ... cache miss judge-request e60f2e7f2bf9`

Source: retry log lines 219–220. The run has 178 `cache hit judge-request` lines and 115 misses. This is direct evidence against cache identity instability for the first 178 identities.

The originally timed-out `cde040…` request succeeded as retry attempt 1:

> `"attempt": 1,`
>
> `"elapsed_seconds": 2.8889541170210578,`
>
> `"outcome": "success",`
>
> `"payload_sha256": "cde04006c77c3e6c33bfae0488f2ae8adeed147a0eac62cd7b1d5491baeed83e"`

Source: `outputs/bsbench-v2/provider-evidence/cde04006c77c3e6c33bfae0488f2ae8adeed147a0eac62cd7b1d5491baeed83e-000001-success.json`. This disproves a second timeout of the original payload.

The later request failed after 114 new successes:

> `"attempt": 115,`
>
> `"elapsed_seconds": 300.56547157495515,`
>
> `"exception_type": "TimeoutError",`
>
> `"payload_sha256": "4e8bc1b07f6ff5e616f7f8c46da07a33ffa9e6ae223bf2f4b038fb2fef3cf4b6"`

Source: `outputs/bsbench-v2/provider-evidence/4e8bc1b07f6ff5e616f7f8c46da07a33ffa9e6ae223bf2f4b038fb2fef3cf4b6-000115-failed.json`.

The cache has a distinct, structural defect: its identity contains metadata outside the provider payload.

> `identity = {`
>
> `    "schema": "bsbench-judge-request-cache-v1",`
>
> `    "request": request,`
>
> `    "judge_model": request["payload"]["model"],`
>
> `    "judge_endpoint": self.endpoint,`
>
> `    "upper_usd": upper_usd,`
>
> `}`

Source: `src/steering_lite/benchmark/adapters.py:135-141`.

For `4e8bc1…`, cache inspection found 16 records with the same raw payload and `request_key` `bbc632…` but different `comparison_id`s. The provider evidence also records duplicate dispatches of that payload at attempts 111, 113 and 115. This is not a cross-retry cache-key instability. It is a lack of payload-level deduplication: `comparison_id` is metadata needed for the response record, but it changes the cached remote-call identity.

## ML-debug form (applicable rows)

| row | answer |
|---|---|
| log length; config | 400-line retry log; Qwen/Qwen3.5-4B generation, DeepSeek `deepseek/deepseek-chat` strict JSON judge, 10 s pacing, 300 s read timeout |
| expected result | only cached generation; final judgments advance from existing 178 responses |
| observed surprise | parent assumption of cde-only work; actual plan has 336 requests, so there were 157 unfinished records after cde recovery |
| null/control | first 178 exact request identities cache-hit; that is the relevant cache-stability control |
| failure number | 300.565 s `TimeoutError` on `4e8bc1…`; provider outcome unknown |
| second cause | provider stalled vs local socket path; callback evidence identifies only `TimeoutError`, so not distinguishable from this record alone |
| absent metric | pre-dispatch count of unfinished *request records* and of unique raw payloads; dry budget only reported broad method upper |
| wall clock | 1,963 s for 115 paced requests, including one 300 s timeout; no GPU stage in retry |

## Hypotheses

### H1 [harness | Almost Certain | 95%]

- **Mechanism:** The retry correctly reused 178 completed per-row cache records, successfully recovered `cde040…`, then continued through still-unpersisted final-judgment records until `4e8bc1…` failed.
- **Evidence:** The log transitions from `cache hit ... cf9993983ba3` to `cache miss ... e60f2e7f2bf9`; provider evidence gives cde attempt 1 success and 4e8bc1 attempt 115 timeout.
- **Contrary evidence:** The prior audit described cde as the only missing cache key. That description did not count the complete 336-record plan.
- **Discriminating test:** Build the complete requests list from cached f054 and count rows with exact cache files before dispatch. It reports 292 persisted successful cache records, one terminal-but-uncached timeout, and 43 untouched records (44 cache misses remain).
- **Fix/action:** Add that count to the cache-aware preflight before any resume.
- **Interpretability:** partial; completed response records are usable, but KV final judgment is incomplete.

### H2 [bug | Almost Certain | 95%]

- **Mechanism:** Judge-response caching keys on metadata-bearing `request`, rather than the endpoint and canonical raw payload. Identical OpenRouter calls can be paid repeatedly when only `comparison_id` differs.
- **Evidence:** `LocalJudgeAdapter` puts `"request": request` in identity; cache enumeration found 16 `4e8bc1…` records with one raw request key `bbc632…` and different comparison IDs.
- **Contrary evidence:** Separate calls could be intentional independent samples. The model is requested with `temperature: 0`, and no plan states that per-row duplicate sampling is intended.
- **Discriminating test:** Offline: construct two request records with same payload but different comparison IDs, run the cache callback twice, and verify only one callback executes while each metadata row can reference the response.
- **Fix/action:** Cache the remote response by canonical `{endpoint, request_key, judge_model, upper_usd}`; then attach each caller's own metadata outside the remote cache record. Do not mutate existing response records as though they had been deduplicated.
- **Interpretability:** yes for existing completed records; no claim about a cache-only rerun until payload-level cache behavior is validated.

### H3 [provider | Likely | 65%]

- **Mechanism:** The failure is an OpenRouter/route response stall rather than a schema or local parsing fault.
- **Evidence:** Callback boundary persisted `TimeoutError` after 300.565 s; the process trace fails inside `ssl.py ... self._sslobj.read`.
- **Contrary evidence:** No provider request ID, body, or invoice can attribute the timeout, so a local network-path failure remains plausible.
- **Discriminating test:** A parent-approved retry of only the exact `4e8bc1…` payload after the unfinished-request count is made explicit. Success would weaken persistent-provider-failure; another timeout would raise it.
- **Fix/action:** No retry under this worker without parent direction; retain 300 s timeout and exact payload/schema if approved.
- **Interpretability:** partial; no judge response exists for the failed record.

### H4 [accounting | Almost Certain | 95%]

- **Mechanism:** The initial receipt import used decimal shorthand `0.0022644`, which is one binary float ULP above the stored `0.0022643999999999997` reservation, falsely emitting an overage after it had written settlement.
- **Evidence:** Failed import log ends `actual cost $0.0022644 exceeded reservation $0.0022643999999999997`; the reservation itself stores the longer value.
- **Contrary evidence:** There is no external invoice to establish an actual amount above or below upper; therefore upper treatment is intentionally conservative.
- **Discriminating test:** Exact stored upper records settlement without overage.
- **Fix/action:** Use the stored reservation upper verbatim for estimate receipts. Preserve failed import evidence; do not treat it as a price change.
- **Interpretability:** yes; ledger is now conservative with no active unresolved reservation.

## Decision

**Resolve-condition verdict:** not met. The requested KV final judgments require all 336 planned records; 292 successful responses are persisted, one later record has only an unknown timeout outcome, and 43 records are untouched. The final-stage cache has not been persisted.

**Prediction check:**

| prediction | result |
|---|---|
| all earlier per-request caches are reused | supported for 178 exact identities |
| cde040 retry fails again | contradicted: it succeeded in 2.889 s |
| no additional requests after cde040 | contradicted: 114 later successes and one later timeout |
| cache key stable across retry | supported for 178 exact cache identities |

**Earliest unsupported link:** Preflight did not calculate complete final-judgment request completion or unique payload counts. That is why it could not distinguish a one-request recovery from 157 unfinished request records.

**Validity:** KV final-evaluation result is incomplete, not negative. Probability that an aggregate KV final claim is invalid if reported now is approximately 0.95–0.99, because the final stage lacks persisted complete judgments.

**Missing evidence ranked:** (1) an exact request-record completion count and unique-payload mapping; (2) provider-side attribution for `4e8bc1…`; (3) a no-paid local test of payload-level cache behavior.

**Recommended sequence:** Keep VJP stopped. First give parent the counted diagnosis and this audit. If parent directs repair, make the payload-level cache contract explicit and demonstrate it offline; then calculate the remaining unique-payload bound and require a new <$50 preflight. Only then retry the currently affected `4e8bc1…` work, preserving its strict schema/payload and 300 s timeout. Do not claim KV completion until the complete final-judgments cache persists.
