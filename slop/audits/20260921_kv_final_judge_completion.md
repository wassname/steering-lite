# KV final-judgment completion audit

- worker: PI[gpt-5.6-terra]
- run: `slop/verification/20260921T151100Z_kv-final-judge-fourth-retry.log`
- code revision before the run: `772c231580cbeb3ae0e1340b45137e590a315f4b`

## Observation

The fourth unchanged KV retry exited 0. The saved KV-only run summary has 84 final answers and health records: 20 numbered evaluation prompts at each of the three predicted evaluation doses, plus four disjoint transfer cases at each dose. Its final judgment records are complete:

| record type | observed | required by saved plan |
|---|---:|---:|
| final answers | 84 | 84 |
| health records | 84 | 84 |
| aware judgments | 168 | 168 |
| blind judgments | 168 | 168 |
| provider request / response records | 336 / 336 | 336 / 336 |

Source: `outputs/bsbench-v2/run-summary.json`, `conditions.kv_cache_gram.final` and `final_judgments`.

The retry reserved and settled 39 remaining DeepSeek judgment calls, with `$0.0111368734` recorded actual cost under a `$0.0780156` reservation upper. The previous 43-miss boundary comprised the prior retry's four recovered requests plus these 39 calls. The cache identity and request multiplicity were unchanged.

Source: `outputs/bsbench-v2/costs.jsonl`, reservations from `2026-09-21T15:11:00Z` onward.

The ledger has no `unresolved` or `overage` event. It retains 26 earlier `estimated_at_reservation_upper` records, mostly Modal work and prior judge timeouts; these are conservative settled accounting, not active remote work. The fresh cache-aware preflight reports `$28.8003012526` total upper, including `$17.8894152526` existing ledger commitment and `$2.00` external commitment.

Source: `outputs/bsbench-v2/costs.jsonl`; `outputs/bsbench-v2/cost-estimate.json`; `slop/verification/20260921T152100Z_post-kv-vjp-delta-dry-preflight.log`.

## Decision

KV-cache-Gram is terminally complete and its final data can be reused. The remaining VJP-delta and VJP-cache work is below the approved $50 bound. Continue with VJP-delta only; do not dispatch VJP-cache until VJP-delta has reached a terminal, reconciled state. -- PI[gpt-5.6-terra]
