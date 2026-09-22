# Random calibration return rejected locally

PI/OpenAI; 2026-09-22. Run `proc_f2df`, source `5972405`, branch `rewrite/bsbench-vjp`. No paid dispatch in this repair.

| stage | expected | observed | expected? | evidence | missing | consequence |
|---|---|---|---|---|---|---|
| prompting judgments | strict aware/blind/persona responses | 80/40/12 exact reusable responses | yes | fresh preflight below | none for reuse | no repeated judge cost |
| extraction | seed 0, layers 7/11/15/19/23 | vector metadata matches | yes | vector metadata below | original in-memory config | vector preserved |
| candidate generation | returned signed candidates pass validation | backend returned; local config validation raised | no | local and Modal logs | complete returned candidate payload | no usable candidate cache |
| final generation/judging | only after validated candidates | not dispatched | yes | local traceback | final outcomes | no method-effect conclusion |
| recovery | retrieve retained complete return | Modal returned NotFoundError | no | retrieval log | full candidate answers/health | cannot reconstruct candidate cache |
| accounting | all reservations resolved | failed GPU estimated at original $0.8646938667 upper | yes | reconciliation log | provider invoice | conservative estimate, not actual billing |

## Evidence and cause

The complete local log (227 lines) ends with:

> ValueError: calibration backend method_config does not attest to the signed method specification

[Local log](../verification/20260922_corrected-five-seed-modal-sweep-entrypoint.log).
The complete remote log records extraction and repeated four-question generations from 19:35:41 to 19:39:04 +08, then:

> Stopping app - uncaught exception raised locally: ValueError('calibration backend method_config does not attest to the signed method specification').

[Modal log](../verification/20260922_random-attestation-modal.log), app `ap-iXzGz6eid8rpUufI1sLTnP`, call `fc-01M34E94Y7339VCGHP7X3NG39F`.

The saved vector hash is `b82951205da0f602652138c1491b0368a30e6d109a3609711bee5d909ce2269f`. Its safetensors metadata has `method=random`, ordered layers `[7,11,15,19,23]`, `seed=0`. [Metadata](../verification/20260922_random-returned-vector-metadata.json). This JSON does not itself prove the original tuple representation. Actual `pipeline.method_config(...).to_dict()` returns a tuple; the production regression now exercises that factory result through candidate settlement, final generation and immediate JSON-cache reread.

1. Representation mismatch (bug, almost certain, 99%): actual factory returns tuple; signed spec uses list; previous equality rejected equal contents. Sidecar metadata supports matching semantic values. The regression passes after normalizing only ordered tuple/list representation. Wrong order, numeric element type, seed, VJP target and skip still reject.
2. Wrong extraction configuration (bug, remote, 2%): could independently exist, but preserved metadata matches exact requested method/layers/seed. Complete return is unavailable; do not infer its other fields from the vector alone.
3. Method ineffectiveness or generation damage (method, unresolved): neither explains the local type comparison. No candidate answer/health payload survived, so no probability of steering success is inferred from progress lines.

Full result recovery failed with:

> modal.exception.NotFoundError: No Function Call with ID 'fc-01M34E94Y7339VCGHP7X3NG39F' found.

[Read-only retrieval](../verification/20260922_random-modal-return-retrieval.log). No candidate records were fabricated from progress logs or the surviving vector. Side-by-side candidate/control text and dose-health metrics are missing; they require the corrected retry. The extraction demo does show identical user/suffix content with only the persona text changed; that checks prompt pairing, not steering effect.

## Repair and verification

- Normalize `layers` tuple/list representation only; preserve order and exact integer values/types. No method/config relaxation.
- Preflight reconstructs direct requests from validated generation caches and uses the same judge-cache identity as execution, including endpoint, routing/max_price and retry policy. It subtracts only present, full-schema-valid responses. Request content/pricing policy is unchanged.
- Reservation `7b44bbd07c9e5da2752e3933b462e4a592f68fd6747e261d3fe2cb580aa386d0` conservatively resolved at its original upper; [reconciliation](../verification/20260922_random-attestation-reconciliation.log). Canonical committed: $17.78465697336667; no unresolved reservations/overage.
- [Production regression](../verification/20260922_random-config-production-proof.log): 7 passed. [Bounded combined checks](../verification/20260922_random-repair-cache-tests.log): 45 passed, 4 deliberately deselected. No exhaustive suite or paid test.

## Budget decision

[Fresh preflight](../verification/20260922_random-attestation-preflight.json), [two arithmetic cases](../verification/20260922_random-retry-budget-cases.json). Both include all probes, the failed calibration upper and external $2:

- Remaining: 20 GPU stages, 8,640 aware requests, 2,400 blind requests; 132 direct requests already cached.
- With another future largest-stage retry reserve: **$50.42898817336667**, fails current strict <$50 policy.
- Original single-stage retry allowance consumed, without allocating a second future reserve: **$49.5642943067**. Parent selected this case in Intercom seq13 (2026-09-22 11:51:54 UTC), quoted below.

Resolve condition “validated signed candidate return” was not met. Prediction that correctly configured returned vectors pass validation was contradicted by tuple/list handling; steering-effect predictions remain unresolved. Earliest unsupported step is complete candidate artifact acceptance. Treat this as an invalid/incomplete method result, not a negative benchmark finding (probability it cannot support a method comparison: >99%). Highest-information evidence: actual config-factory tuple reproduction; matching sidecar metadata; local rejection after remote generation. A valid retry with full candidate artifact and final judgments would change that verdict.

Parent decision (Intercom seq13):

> use the ORIGINAL one-stage retry allowance, now consumed by reconciled reservation 7b44bbd07c9e5da2752e3933b462e4a592f68fd6747e261d3fe2cb580aa386d0. Do not automatically replenish it.

> A further paid-stage failure stops dispatch for review/rebudget, not automatic renewed allowance.

Implemented only for that exact reconciled reservation, matching kind and original upper. Missing resolution or mismatched evidence rejects. No general no-reserve flag exists. Committed cost is unchanged; the preflight explicitly names the original allowance, consumed reservation and zero remaining reserve. Aggregate <$50 and per-attempt reservation checks remain. The original fresh arithmetic above remains saved; [final authorized-policy preflight](../verification/20260922_random-retry-final-preflight.json) records the selected case. [Final focused checks](../verification/20260922_random-repair-final-tests.log).

Next: parent resumes the same full entrypoint after inspection; it starts at the failed random stage after direct cache reuse. No timeout/token cap or scientific scope changes. This worker does not launch paid work.
