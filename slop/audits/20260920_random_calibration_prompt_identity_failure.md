# Random calibration prompt identity failure

Target: the first real vector stage of the BS-bench sweep, `random` / `calibration-candidates`, on `rewrite/bsbench-vjp` at `ec94f41`.

## Stage table

| stage | expected | observed | expected? | clues | missing metric | consequence |
|---|---|---|---|---|---|---|
| cached bare/prompting | production cache reuse | both generation stages and the 80 prior judge responses were cache hits | yes | `slop/verification/20260920T015849Z_real-full-sweep-429-recovery.log` | none | no duplicate generation before the vector stage |
| persona recovery | 12 saved persona judgments after the one prior 429 was reconciled | 12 paced requests settled; `persona-validation` cache saved | yes | recovery log, lines 45–59 | aggregate persona result is not yet interpreted | the previous 429 was not repeated |
| random Modal calibration | one serialized candidate response per coefficient × 4 calibration prompts | Modal function ran through baseline plus seven four-prompt candidate generations | yes | `slop/verification/20260920T020657Z_modal-random-calibration-app-identifiers.log` | returned candidate JSON was not persisted before local validation | provider work completed, but the local caller rejected it |
| local candidate validation | exactly matching `(coefficient, prompt_index, prompt_sha256)` keys | `ValueError: candidate items must cover exactly every coefficient and calibration prompt once` | no | recovery log traceback; `src/steering_lite/benchmark/production.py:98-110` | the discarded returned payload | later methods stopped; reservation is unresolved pending reconciliation |

## Primary evidence

The Modal provider log identifies the completed function execution:

> `2026-09-20 10:01:10+08:00 fu-ZiGBwS8F6TsPPU8ygX1fKt fc-01M2Y8Q0X2PQK31M4A4CVEPF78 ta-01M2Y8Q14CX8HXKJPRWFQ0DMYR Warning: You are sending unauthenticated requests to the HF Hub.`
>
> `2026-09-20 10:04:23+08:00 fu-ZiGBwS8F6TsPPU8ygX1fKt fc-01M2Y8Q0X2PQK31M4A4CVEPF78 ta-01M2Y8Q14CX8HXKJPRWFQ0DMYR 2026-09-20 02:04:23.691 | INFO | steering_lite.benchmark.generation:generate:82 - generation 4/4`
>
> `2026-09-20 10:04:24+08:00 Stopping app - uncaught exception raised locally: ValueError('candidate items must cover exactly every coefficient and calibration prompt once').`

Source: Modal `ap-4ublFhNSLuYc5l5rDUA40J`, function `fu-ZiGBwS8F6TsPPU8ygX1fKt`, call `fc-01M2Y8Q0X2PQK31M4A4CVEPF78`; captured at `slop/verification/20260920T020657Z_modal-random-calibration-app-identifiers.log`.

The timing (about 194 seconds), extraction trace, and eight `generation 1/4` groups establish that the provider function completed baseline generation plus seven candidate doses. The final exception is explicitly local, after the remote call returned. The provider log does not provide a billable receipt, so the ledger must keep the conservative reservation upper.

The mismatch is deterministic from the callback hash and the validator's required hash:

- The callback makes `calibration_prompts = canonical_prompt_texts(tokenizer, prompts, config["prompt_spec"])`, then `_candidate_policy` stores `sha256(prompt)` for those canonical chat strings: `scripts/run_bsbench_modal.py:189-207`.
- The consumer expects `sha256(prompt)` for the four original source prompts: `src/steering_lite/benchmark/production.py:98-110`.
- The local reproduction uses the production `canonical_prompt_texts` and `_candidate_items` functions against BSV2-001 through BSV2-004. All four raw versus canonical hashes differ; BSV2-001 is raw `3e640783…` versus locally serialized `fab26937…`. It constructs the seven-dose set shown by the provider logs, reproduces the exact production `ValueError`, then succeeds after replacing only each item's hash with the source-prompt hash. Evidence: `slop/verification/20260920T021301Z_random-calibration-prompt-key-repro.log`.

Inference: the observed key set contains seven coefficients × four prompt indices = 28 keys with canonical prompt hashes. The expected set contains the same coefficient/index pairs with source-prompt hashes. The local reproduction measures 28 missing expected keys, 28 extra observed keys, and zero duplicates; this is a prompt-identity mismatch, not a coefficient/search-coverage mismatch. The seven coefficients are inferred from baseline plus seven candidate generation groups and the doubling policy; the discarded response prevents direct inspection of the returned coefficient list.

## Hypotheses

### H1 [bug | Almost Certain | 95%]

- Mechanism: the Modal callback hashes the model-input chat serialization, while the local validator hashes the source prompts.
- Evidence: the two code paths above and all four distinct hash pairs in the local reproduction.
- Contrary evidence: the unpersisted returned payload could expose another malformed field, but it cannot make the two hash definitions equal.
- Discriminating test: a regression invokes `_candidate_policy` on serialized prompts while supplying source hashes; expected result: items carry source hashes and `_candidate_items` accepts them.
- Fix/action: pass the source prompt hashes separately to `_candidate_policy`; do not weaken exact coverage.
- Interpretability: no experiment result is interpretable from this stage, but the failure mechanism is interpretable.

### H2 [measurement | Highly Unlikely | 5%]

- Mechanism: a provider response had missing/duplicate candidate rows independently of the hash mismatch.
- Evidence: the consumer only reports aggregate key-set failure and the returned payload was discarded.
- Contrary evidence: `_candidate_policy` uses `zip(..., strict=True)` and emits one item per generated answer, while the provider log shows complete four-prompt groups.
- Discriminating test: the fixed retry must save the full response; compare returned rows to the expected set.
- Fix/action: retain existing exact validation and inspect the saved retry artifact.
- Interpretability: partial until the retry result is saved.

## Decision

The provider function completed, but no provider billing receipt is available. Reconcile reservation `2f2ae5cdc97c66878e73a5e3189a87926118ad64cf358aee2cae32c211f3d9f7` at its conservative upper `$0.884346`, not `$0`. Apply only the source-prompt-hash fix, run focused and full offline tests, then rerun the dry budget. Retry only `random` calibration after the budget remains strictly below `$50`; do not dispatch later methods until the fixed stage persists and settles.

-- PI/OpenAI
