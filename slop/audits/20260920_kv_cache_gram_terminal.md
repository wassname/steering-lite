# KV-cache-gram 300-second recovery audit

Target: the authorized KV-only recovery. Provenance: `13d084f` at launch; command and the complete 188-line log are in [`20260920T054711Z_kv-cache-gram-300s-recovery.log`](../verification/20260920T054711Z_kv-cache-gram-300s-recovery.log). The process exited 0 after 1,363 seconds. No VJP method was started.

-- PI[gpt-5.6-terra]

| stage | expected | observed | expected? | clues | missing metric | consequence |
|---|---|---|---|---|---|---|
| KV candidate extraction | Reuse prior Modal candidate | reused 36 candidate items | yes | candidate cache `da7468f825a1`; reservation `a900…` | independent vector quality check | no new Modal extraction cost |
| Judge cache recovery | Reuse 15 records then retry the timeout position | 15 hits; first fresh key `30e2ad75b2b2`, matching old timed request kind | yes | reconciliation JSON | provider request ID | old timeout is no longer an unknown outcome |
| Judge requests | 300s maximum read; 10s minimum callback starts | 129 successes, 0 failures; minimum start interval 10.000028s | yes | sanitized timing evidence | provider-side receipt ID | no new unresolved reservation |
| Candidate evaluation | measured useful, coherent dose, or terminal condition | all 9 doses have negative `dose_score`; all incoherent | yes for terminal path | terminal artifact | independent human or second-judge ratings | final generation correctly prevented |
| Ledger | every recovery reservation paired | 129 reserved, 129 settled; `$0.0466293941` actual | yes | reconciliation JSON | exact aggregate provider attribution not needed here | no open unresolved item |
| Modal lifecycle | no active task after cached-only run | recovery app stopped with 0 tasks | yes | `.venv/bin/python -m modal app list --json` | none | no active Modal work |

## Primary evidence

The runner records the actual command and cache reuse:

> `command=just sweep "--run --backend real --method kv_cache_gram --openrouter-read-timeout 300"`
>
> `2026-09-20 13:48:04.901 | INFO | steering_lite.benchmark.cache:cached:35 - cache hit calibration-candidates da7468f825a1`
>
> `2026-09-20 13:48:05.081 | INFO | steering_lite.benchmark.cache:cached:37 - cache miss judge-request 30e2ad75b2b2`

Source: [`20260920T054711Z_kv-cache-gram-300s-recovery.log`](../verification/20260920T054711Z_kv-cache-gram-300s-recovery.log), written by the recovery runner. This establishes local cache behavior, not an external provider receipt.

The first replacement request is the earlier timed payload. Its new timing evidence says:

> `"payload_sha256": "94a91f91f87a3999c731952cd7d0114c54fb30894aba93d1f93cbd6cdc39bf38"`
>
> `"dispatch_started_at": "2026-09-20T05:48:05.083265+00:00"`
>
> `"response_finished_at": "2026-09-20T05:48:06.989391+00:00"`
>
> `"elapsed_seconds": 1.9061242070165463`

Source: [`94a91…-000001-success.json`](../../outputs/bsbench-v2/provider-evidence/94a91f91f87a3999c731952cd7d0114c54fb30894aba93d1f93cbd6cdc39bf38-000001-success.json), written at the callback boundary after the response. It directly establishes only local start/finish timing.

The terminal record says:

> `"status": "no measured useful, coherent dose"`
>
> `"selection_predicate": "useful is directed_effect - 4 * off_target_effect > 0; coherent is no health reasons and max steered off-axis <= 2.5"`
>
> `"final_dispatch_prevented": true`

Source: [`run-summary.json`](../../outputs/bsbench-v2/run-summary.json), produced by the completed runner. The nine measured `dose_score` values range from `-3.3500000000000005` to `-20.575`; none is positive.

A complete raw candidate shows the low-dose vector did not visibly collapse generation:

> `"coefficient": 0.1`
>
> `"response": "The decomposition should isolate indemnity liabilities by specific IP asset class and jurisdiction to accurately reflect the target's material concentration risks. This granular approach ensures the purchase price allocation and escrow holdbacks are calibrated precisely to the unique exposure profile of the SaaS business."`

At high dose it did collapse:

> `"coefficient": 25.6`
>
> `"response": "I am the current. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the. I am the"`

Source: KV `candidate_items` in [`run-summary.json`](../../outputs/bsbench-v2/run-summary.json). These are two selected examples: low and maximum coefficient, not a representative sample.

## Ledger and timing reconciliation

[`20260920T061041Z_kv-cache-gram-terminal-reconciliation.json`](../verification/20260920T061041Z_kv-cache-gram-terminal-reconciliation.json) pairs every recovery reservation by ID. It records 129 new judge reservations and 129 settlements, `$0.2586456` reserved upper cost, `$0.0466293941` actual cost, and an empty open-unresolved list. All 129 new callback records report success. The start-to-start minimum is 10.000028 seconds; no interval is below 10 seconds. Successful elapsed times range from 1.774 to 25.670 seconds, far below the new 300-second limit.

The recovery created Modal app `ap-DtsbAEMYHVUDS63rq0kNsJ` as the runner lifecycle entered, but its app listing says `"state": "stopped"` and `"tasks": "0"`. The completed candidate cache meant no new candidate extraction task was needed.

The paid-disabled post-run preflight has `"total_upper_usd": 46.2553782696` under the `$50` limit, while `"paid_execution_enabled": false`. This is a planning bound, not authorization for VJP work.

## ML-debug form, abbreviated for this terminal calibration

| row | answer |
|---|---|
| log/config | 188 lines; exact command used the unchanged model/endpoint/payload and `--openrouter-read-timeout 300` |
| stated expectation | reuse candidate and 15 records, resolve the first old timeout, then stop on terminal calibration or failure |
| observed timing | 129 success records; 1.774–25.670s response elapsed; minimum start interval 10.000028s |
| control / null | no method-effect null is available inside this KV-only calibration; usefulness null is `dose_score <= 0` from the recorded predicate |
| qualitative sample | low-dose response is fluent but accepts the fabricated premise; highest-dose response is repetitive, quoted above |
| missing metric | independent human/second-judge ratings and a holdout calibration cohort |
| wall clock | 1,363 seconds, dominated by 10-second paced judge calls; no new GPU task |

## Hypotheses

### H1 [method | Highly Likely | 80%]

- **Mechanism:** On this four-prompt calibration cohort, KV-cache-gram did not produce a dose that simultaneously has positive directed-minus-four-times-off-target effect and passes coherence.
- **Evidence:** The terminal record says `"no measured useful, coherent dose"`; all nine `dose_score` values are negative.
- **Contrary evidence:** This is one cohort and one judge model; a favorable dose could exist outside the tested grid or under independent ratings.
- **Discriminating test:** A separately authorized holdout calibration with an independently designed judge or human subset. A positive coherent score would weaken this local negative.
- **Fix/action:** Do not generate a final KV result from this calibration; preserve its terminal condition.
- **Interpretability:** yes for the tested calibration predicate; partial for the general method.

### H2 [harness | Likely | 70%]

- **Mechanism:** The previous request was an isolated read-timeout after provider acceptance, rather than a deterministic malformed payload or systematic latency problem.
- **Evidence:** The exact timed payload completed in 1.906 seconds during the 300-second recovery, and all 129 replacement requests completed in at most 25.670 seconds.
- **Contrary evidence:** The old call had a 181.610-second timeout; a future tail event can still exceed 300 seconds.
- **Discriminating test:** Continue writing callback timing on each authorized run. Another long tail near 300 seconds would support a provider/harness tail rather than an isolated event.
- **Fix/action:** Keep the 300-second retry parameter only for this recovery class; retain upper-cost settlement if a future response is unknown.
- **Interpretability:** yes for the completed retry; partial for provider billing without request-level provider IDs.

### H3 [measurement | Plausible | 40%]

- **Mechanism:** The judge may rate premise acceptance and off-axis damage differently from a human evaluator, changing the sign or coherence outcome.
- **Evidence:** The terminal depends on a single `deepseek/deepseek-chat` judge and only four calibration prompts; the low-dose raw example is fluent while the judge-derived score is not useful.
- **Contrary evidence:** 144 complete candidate judgement responses were persisted with AB/BA order, and the high-dose raw repetition matches the health failure.
- **Discriminating test:** Blind human or independent-model scoring of a prespecified stratified sample at 0.1, 0.2, and 12.8. Agreement would support the current metric; disagreement localizes a measurement issue.
- **Fix/action:** Treat the result as a calibration-terminal result, not a broad quality claim.
- **Interpretability:** partial.

## Decision

1. **Resolve-condition verdict: met.** The authorized recovery completed without failure or unknown outcome; the terminal record prevents final generation because no measured useful, coherent dose exists.
2. **Prediction check:** cache reuse, first timed request recovery, ≥10-second callback starts, and paired reservations were all supported. No prediction of KV effectiveness was recorded before the run.
3. **Earliest unsupported link:** whether this judge/cohort measures the intended persona axis as a human would. Independent ratings would test it.
4. **Validity:** invalid would mean a missing response, unpaired ledger item, changed candidate/payload, or pacing violation. None occurred. `P(this local terminal result is invalid) ≈ 0.15–0.30`; classify it as a credible local negative, not a general-method conclusion.
5. **Highest-information clues:** (a) 129/129 successful, settled requests removes the recovery failure confound; (b) 10.000028-second minimum dispatch interval directly validates pacing; (c) all nine negative dose scores, plus high-dose repetition, supports the local terminal condition.
6. **Missing metrics:** independent ratings, holdout prompts, then provider request IDs.
7. **Bugs requiring code changes:** none identified in this completed recovery.
8. **Misconceptions requiring reinterpretation:** do not interpret ledger reservation intervals as callback dispatch pacing; use the new callback evidence.
9. **What changes the verdict:** a complete independent calibration with at least one useful, coherent dose would overturn the local terminal conclusion.
10. **Recommended sequence:** wait for review; retain the KV terminal cache and do not start `vjp_delta` or `vjp_cache` until explicitly authorized.
