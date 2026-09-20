# PCA terminal calibration audit

Target: authorized PCA-only continuation after the interrupted judge request. It completed with no useful, coherent dose, so the harness did not run final PCA generation or a later method.

Provenance: [`20260920T050323Z_pca-authorized-recovery-run.log`](../verification/20260920T050323Z_pca-authorized-recovery-run.log) is the complete 140-line process log (one final JSON line is large). Its executed summary is ignored at `outputs/bsbench-v2/run-summary.json`, with source identity hash `8ec996eba103e978d614735e253ad4e34996e79bb55b558ab53d58ef7d19c304`. The log does not record a Git revision, so revision provenance is partial. No source file changed during this run.

-- PI[gpt-5.6-terra]

| stage | expected | observed | expected? | clues | missing metric | consequence |
|---|---|---|---|---|---|---|
| PCA candidates | Reuse completed Modal candidate extraction | Cache hit `91e3f8bfbfae` | yes | recovery log line 38 | Modal invoice | No new Modal call |
| Candidate judge records | Reuse 23 persisted records, then continue at `7d5c…` | 23 cache hits; first fresh reservation is `judge-7d5c…` | yes | reconciliation JSON | Provider request IDs | 73 new requests, not a fresh 96-request run |
| Spending | Every new request settles with a receipt | 73/73 settled, `$0.027992505`, no open unresolved reservation | yes | reconciliation JSON | Account-level metadata after run | No paid work remains active |
| Dose selection | At least one dose has positive directed effect after off-target penalty and is coherent | None does | no | terminal result | Repeated seeds / another judge | PCA stops before transfer/final evaluation |
| Final PCA stages | Run only after a selected calibration dose | `final_dispatch_prevented: true` | yes | terminal result | Held-out score | No result for later stages |

## Evidence

The recovery log records:

> `cache hit calibration-candidates 91e3f8bfbfae`
>
> `cache miss candidate-judgments dc1575e78cd9`

The latter is the aggregate cache, absent because the interrupted run had not reached its final save. The log then has 23 `cache hit judge-request` records. The first new ledger reservation is:

> `"kind": "judge-7d5c703c718b4020c818b27f825fb0190b540bee6d58600a83f41155286da7f6"`

[`20260920T051851Z_pca-terminal-cache-ledger-reconciliation.json`](../verification/20260920T051851Z_pca-terminal-cache-ledger-reconciliation.json) records 73 new reservations, 73 settlements, actual total `$0.027992505`, and an empty `open_unresolved_reservations` array.

The terminal result records the selection rule:

> `useful is directed_effect - 4 * off_target_effect > 0; coherent is no health reasons and max steered off-axis <= 2.5`

The six measured doses are:

| coefficient | directed effect | off-target effect | score | coherent | useful |
|---:|---:|---:|---:|---|---|
| 0.1 | -0.5625 | 0.4375 | -2.3125 | yes | no |
| 0.2 | -0.3000 | 0.9875 | -4.2500 | yes | no |
| 0.4 | -0.5250 | 1.7750 | -7.6250 | no | no |
| 0.8 | -0.7625 | 0.7000 | -3.5625 | yes | no |
| 1.6 | 1.9125 | 2.3875 | -7.6375 | no | no |
| 3.2 | -2.2000 | 3.5250 | -16.3000 | no | no |

At `3.2`, generation health also records `unfinished`, `role_leak`, and `repetition`. The summary terminal status is:

> `no measured useful, coherent dose`

## ML-debug form

| row | answer |
|---|---|
| Log and executed config | 140 lines; PCA only, Qwen/Qwen3.5-4B, DeepSeek judge, default endpoint, 10-second pacing, 180-second read timeout. |
| `SHOULD:` lines | None in the run log. The executable selection predicate is quoted above. |
| Null/control scale | No null distribution, seed replicate, or alternate judge was run. The reported score only establishes this candidate set under this judge. |
| Complete sample | At coefficient `3.2`, health saw 4/4 unfinished responses, 3 role leaks, and max repetition `0.9047619`; raw pair outputs are in the run summary. |
| Baseline / held-out | Not reached because the terminal calibration condition stopped final dispatch. |
| Surprising evidence | `1.6` has positive directed effect `1.9125`, but off-target `2.3875` makes its score negative. This separates directional movement from a usable intervention. |
| Missing evidence | Replicates, an independent judge, a calibration cohort larger than four prompts, and actual Modal invoice data. |
| Fresh review | Requested `reviewer-anthropic`; it failed before reading due provider 429 `credits_required`. This is infrastructure evidence only; it was not retried. |
| Wall time | Process elapsed 792 seconds. Modal candidate work was reused; 10-second judge pacing dominated the 73 fresh calls. |

## Hypotheses

### H1 [method | Likely | 70%]

- **Mechanism:** PCA moved the target direction at 1.6 but produced larger penalized off-target behavior, so this calibrated vector has no usable dose under the stated rule.
- **Evidence:** The terminal record gives `1.6: directed_effect 1.9125`, `off_target_effect 2.3875`, `dose_score -7.6375`, `useful false`.
- **Contrary evidence:** Four calibration prompts and one judge cannot estimate the method's broader behavior precisely.
- **Discriminating test:** An independently authorized repeat with another judge and a new held-out calibration cohort. A useful coherent dose in both would contradict this run-specific negative.
- **Action:** Do not transfer or final-evaluate this PCA vector. Keep the cached candidate and judge records for audit.
- **Interpretability:** partial.

### H2 [measurement | Plausible | 35%]

- **Mechanism:** The judge's on-axis/off-axis ratings, not the generation health detector, reject 0.4 and 1.6; judge noise or criterion calibration could determine the result.
- **Evidence:** At 0.4 and 1.6, `generation_health.reasons` is empty, while `coherent` is false under the maximum steered off-axis threshold.
- **Contrary evidence:** The same judge's blinded comparisons and the fixed selection predicate were used consistently across all six doses.
- **Discriminating test:** Score the persisted response pairs with an independent pre-specified judge. Agreement on off-target rank/order supports the current conclusion.
- **Action:** Treat the result as a terminal calibration negative, not a proof that PCA steering is impossible.
- **Interpretability:** partial.

### H3 [data | Plausible | 30%]

- **Mechanism:** The four-prompt calibration set is too narrow and includes failures that make the observed directional effect unstable.
- **Evidence:** The candidate calibration case has only BSV2-001 through BSV2-004; at 3.2 all four outputs fail health checks.
- **Contrary evidence:** Lower doses are coherent but still have non-positive directed effects, so sample size alone does not explain all failures.
- **Discriminating test:** Evaluate a separately specified calibration cohort with the same vector and dose grid after review.
- **Action:** Do not change the method from this result alone.
- **Interpretability:** partial.

## Decision

- **Resolve condition:** met for the bounded recovery. The terminal record says `final_dispatch_prevented: true` and `no measured useful, coherent dose`; no later method was launched.
- **Prediction check:** pre-run prediction was that the completed candidate cache and 23 earlier judge records would be reused, with `7d5c…` first. This is supported by the log and reconciliation record. No prediction was recorded for a successful dose.
- **Earliest unsupported link:** a PCA candidate must produce a useful, coherent calibration dose. The measured records do not support that link.
- **Validity:** credible terminal calibration negative for this vector, dose grid, four-prompt cohort, and judge. It is not a broad negative result for PCA steering. Estimated probability that the terminal status itself is invalid: 15%; probability that it would change under a broader calibration/judge protocol is higher and unmeasured.
- **Remaining conservative upper:** `$45.3173758528`, `$4.6826241472` below `$50`, from the paid-disabled post-run preflight. The ledger has no open unresolved reservation.
- **Next sequence:** wait for review. Do not run another paid method. If further investigation is authorized, use the persisted PCA records to determine whether a broader cohort or independent judge is worth its additional cost before generating another vector.
