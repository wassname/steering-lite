# VJP-cache OpenRouter 429 audit

Target: the stopped canonical continuation in `slop/verification/20260920T061625Z_vjp-canonical-continuation.log`, started from commit `0fd6e64` with source hash `d2a8eb38…`. The process used Qwen/Qwen3.5-4B, `deepseek/deepseek-chat`, the unchanged endpoint, a 300 s read timeout, and 10 s dispatch pacing.

## Stage table

| stage | expected | observed | expected? | clues | missing metric | consequence |
|---|---|---|---|---|---|---|
| Earlier conditions | Cache reuse only | All preceding conditions logged cache hits | yes | run log lines 6–65 | none | No earlier candidate generation or judge request was repeated. |
| VJP-delta candidate generation | One cached candidate record and seven doses | Cache `738046…` contains 28 candidate items at 0.1–6.4 | yes | reconciliation `stage_state.vjp_delta` | Modal invoice | Candidate generation completed. |
| VJP-delta judgments | Complete AB/BA aware and blind judgments | Cache `18691e…` has 112 responses; none useful+coherent | yes | `observed_dose_scores` in reconciliation | More calibration prompts/seeds | Local terminal branch correctly avoided final generation. |
| VJP-cache candidate generation | One cached candidate record | Cache `8e8bff…` contains 28 candidate items | yes | reconciliation `stage_state.vjp_cache` | Modal invoice | Candidate work completed and is reusable. |
| VJP-cache candidate judgments | Complete aggregate cache | 83 successful individual request caches, then HTTP 429 on request payload `69a9…`; aggregate stage absent | no | provider evidence `69a9…-000196-failed.json` | Provider request-level billing receipt | Stop; do not begin final generation or another method. |
| Ledger | Every reservation settled or conservatively estimated | 198 new reservations; 429 reservation `6fa…` was unsettled then settled at its $0.0017496 upper; open set now empty | yes | post-receipt ledger JSON | Invoice attribution for `6fa…` | Accounting is conservative, not a claim of actual billing. |
| Modal | No running worker after local failure | App `ap-2ImYkqIjksdYBymvFwWcea` is stopped with 0 tasks | yes | Modal list JSON | Invoice | No active remote task remains. |
| Budget | Paid-disabled upper below $50 | `$48.095003328 < $50`; `paid_execution_enabled: false` | yes | dry-preflight log | A preflight for a specific retry time | No dispatch is authorized by this audit. |

## Chronology and primary evidence

The complete retained run log is 367 lines. It began by reading cache entries for the six earlier calibration conditions. It then created VJP-delta candidate cache `738046a82eb3`, completed its candidate-judgment cache `18691e91dffa`, and created the VJP-cache candidate cache `8e8bff0f303d`.

The VJP-delta raw records establish a method-specific negative under the precommitted predicate, not a claim about all VJP-delta implementations. Its best observed score was -2.9125 at coefficient 0.4; it was coherent but not useful. The cached candidate output at 0.4 starts:

> `The decomposition should isolate liabilities by specific IP asset class and jurisdiction to accurately reflect the concentration risk inherent in your target.`

This is an observation from `outputs/bsbench-v2/cache/candidate-health/b3967fd…json`; it is not an independent quality measure. The corresponding complete judge cache has 112 responses and the seven scores in the reconciliation artifact.

The first VJP-cache candidate-judgment request was sent after candidate generation completed. Requests maintained the requested pacing: the 196 timing records have minimum dispatch-start interval 10.000033 s and zero intervals below 10 s. The final successful request began at `2026-09-20T07:00:40.450646+00:00`, and the failed request began at `2026-09-20T07:00:52.129278+00:00`.

The sanitized provider record reports the decisive error:

> `"status": 429,`
> `"limit_source": "upstream_provider_shared_pool",`
> `"raw": "deepseek/deepseek-chat is temporarily rate-limited upstream. Please retry shortly…"`

Source: `outputs/bsbench-v2/provider-evidence/69a9b64086c88599852b1a63fad9436a1ede839d5bc7f50c28d84a3626607490-000196-failed.json`. This is OpenRouter's error payload about its selected upstream provider; it establishes rejection, not whether an attempted request was billed.

The failed callback marked reservation `6fa662…` unresolved. The no-generation OpenRouter metadata command returned key usage `$0.268215735`, while all settled judge ledger entries total `$0.273016928`. Because that aggregate difference is -$0.004801193, it cannot isolate this request. The receipt therefore settles the failed request at its full `$0.0017496` upper. This clears the accounting condition without claiming the provider charged that amount.

## ML-debug form

| row | evidence |
|---|---|
| log length and config | 367 lines; command and 300 s/10 s parameters are lines 1–4 of the run log. |
| expected result | VJP-delta terminal if no useful+coherent dose; VJP-cache candidate judgments should be complete. |
| observed control | Earlier conditions cache-hit; VJP-delta has a complete 112-response cache. |
| one complete sample | The quoted VJP-delta coefficient-0.4 answer is from the cached raw candidate record. |
| surprise | A 429 arrived despite 10.000033 s minimum dispatch spacing. explained: provider evidence names the upstream shared pool; client pacing is not a provider-capacity guarantee. |
| missing evidence | Per-request provider billing, VJP-cache's remaining candidate judgments, seed variation, and transfer evaluation. |
| wall time | 2,632 s process wall time; 196 callbacks ran from 06:23 to 07:00 UTC. |

## Hypotheses

### H1 [harness | Highly Likely | 80%]

- **Mechanism:** OpenRouter rejected the request because its selected shared upstream provider lacked temporary capacity.
- **Evidence:** The provider record says `"limit_source": "upstream_provider_shared_pool"` and `"deepseek/deepseek-chat is temporarily rate-limited upstream"`.
- **Contrary evidence:** No provider request-limit header or retry-after value was returned; a local request burst cannot be completely excluded.
- **Discriminating test:** A later same-payload VJP-cache retry with the existing 10 s pacing. A success supports a transient provider limit; another immediate 429 leaves upstream capacity unavailable.
- **Fix/action:** Wait for review, then retry only the affected VJP-cache stage with the existing model, endpoint, payload, pacing, and 300 s timeout.
- **Interpretability:** partial; VJP-delta remains interpretable, but VJP-cache does not yet have complete candidate judgment data.

### H2 [measurement | Likely | 65%]

- **Mechanism:** Aggregate key metadata cannot determine the monetary outcome of the rejected request.
- **Evidence:** Key usage `$0.268215735` differs from settled ledger `$0.273016928`; the receipt records `"Aggregate key metadata is lower than the settled judge ledger total and cannot attribute the failed request."`
- **Contrary evidence:** An HTTP 429 is commonly non-billable, and the response returned in 0.798 s, but neither fact is a per-request receipt.
- **Discriminating test:** Provider request-level billing tied to payload `69a9…`. Zero billed cost would replace the conservative upper with a receipt; a positive cost would confirm it was needed.
- **Fix/action:** Keep the upper settlement until a provider receipt is available.
- **Interpretability:** yes for the accounting bound; no claim about actual cost.

### H3 [method | Likely | 70%]

- **Mechanism:** This VJP-delta vector has no useful+coherent calibration dose on the four fixed BS-bench prompts under the fixed `directed_effect - 4 × off_target_effect` metric.
- **Evidence:** The complete cached scores are all negative: `[-7.675, -3.5875, -2.9125, -3.0125, -9.1, -10.35, -19.35]`.
- **Contrary evidence:** One candidate batch and four prompts do not test another vector construction, seed, or data distribution.
- **Discriminating test:** A separately approved held-out/seed experiment, retaining the metric and cache identity rather than changing the criterion after the result.
- **Fix/action:** Record VJP-delta as a terminal method-specific negative; do not dispatch its final generation.
- **Interpretability:** yes, limited to this vector, model, prompts, and score rule.

### H4 [bug | Unlikely | 25%]

- **Mechanism:** Local pacing or cache identity may have caused avoidable provider pressure or stale work.
- **Evidence:** Timing evidence gives a 10.000033 s minimum start-to-start interval and all source cache identities remain at `d2a8eb38…`.
- **Contrary evidence:** The provider response names an upstream shared pool, and the previous 195 requests completed.
- **Discriminating test:** Reinspect timing records and confirm targeted retry cache hits for the 83 successful VJP-cache requests.
- **Fix/action:** No source edit before a failing pacing/cache check exists.
- **Interpretability:** partial; the 429 is a system failure, not an experimental observation.

## Decision

**Resolve-condition verdict:** not met for VJP-cache. The approved continuation required both VJP methods to finish before an audit and reporting; VJP-cache has only partial candidate judgment data. VJP-delta's local terminal condition is met.

**Prediction check:** Earlier conditions would cache-hit (supported). VJP-delta might terminate locally if no useful+coherent dose exists (supported). VJP-cache candidate judgments would complete (contradicted by HTTP 429).

**Earliest unsupported link:** VJP-cache's full candidate judge set is missing. The next evidence is a complete aggregate `candidate-judgments` cache, not final-generation data.

**Validity:** Define invalid as treating the 429 as an experimental VJP-cache result or treating the conservative settlement as actual billing. Probability either error would invalidate a VJP-cache conclusion is almost certain without the missing judgments. Current classification: inconclusive VJP-cache; credible method-specific VJP-delta negative.

**Three highest-information clues:** (1) the provider's explicit `upstream_provider_shared_pool` reason; (2) 195 success records with 10 s start spacing; (3) VJP-delta's complete 112-response cache and uniformly negative fixed-metric scores.

**Missing metrics:** highest expected information is the remaining VJP-cache candidate judgments; next is provider request-level billing; then held-out VJP-delta evaluation.

**Recommended sequence:** wait for authorization and provider recovery; rerun only `just sweep "--run --backend real --method vjp_cache --openrouter-read-timeout 300"`; check that the candidate cache and 83 successful judge requests cache-hit and that the retry starts with payload `69a9…`; stop again on a new failure or unknown outcome. Do not route to another provider, change model/payload, run final generation, or render results before review.

-- PI[gpt-5.6-terra]
