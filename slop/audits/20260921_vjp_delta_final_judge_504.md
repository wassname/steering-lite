# VJP-delta final-judge HTTP 504 reconciliation

- worker: PI[gpt-5.6-terra]
- failed run: `slop/verification/20260921T152600Z_vjp-delta-final-evaluation.log`
- code revision before dispatch: `1361837ea83e58a4a22cfbac4b178ebf0175080b`

## Observation

VJP-delta final generation completed and was cached with 84 answers and 84 health records. Final judgment then made 175 successful cached DeepSeek request records. Request 176 failed before a response cache was written; 160 planned final judgment records remain untouched.

| final judgment boundary | count |
|---|---:|
| planned request records | 336 |
| cached successful records | 175 |
| failed terminal request | 1 |
| untouched records | 160 |

The failed OpenRouter evidence records HTTP 504, `"The operation was aborted"`, after 6.469 seconds. Its provider metadata preserves a prior upstream DeepInfra 429, while OpenRouter selected StreamLake; it is not a strict-schema or local-code error.

> `"status": 504,`
>
> `"reason": "Gateway Timeout",`
>
> `"previous_errors": [{"code": 429, "provider_name": "DeepInfra", ...}],`
>
> `"provider_name": "StreamLake"`

Source: `outputs/bsbench-v2/provider-evidence/35441d3e67c2c43b8e29a9735bc319632a8c19db6cff34dfa8d403cb57ef06a2-000175-failed.json`.

The associated ledger reservation `8dc4a4…` had upper `$0.0022643999999999997`. No provider usage receipt resolves whether the aborted request charged, so it was settled at that exact original upper, rather than a rounded replacement. After the receipt import, `active_unresolved=[]` and `overage=[]`.

Sources: `slop/verification/20260921T161700Z_vjp-delta-final-judge-504-upper-receipt.json`; `slop/verification/20260921T161800Z_vjp-delta-final-judge-504-upper-settlement.log`; `outputs/bsbench-v2/costs.jsonl`.

The post-settlement paid-disabled dry preflight is `$29.755023332 < $50`. This is a conservative global planning estimate: it does not claim that the partially cached final-judgment stage has no remaining requests.

Source: `slop/verification/20260921T162000Z_vjp-delta-final-judge-504-retry-dry-preflight.log`.

## Decision

VJP-delta remains incomplete solely at final judgment. VJP-cache remains stopped. After a bounded backoff, retry VJP-delta unchanged: Qwen3.5-4B, DeepSeek judge, strict schema, 10-second pacing and 300-second timeout. The retry will reuse the cached final generation and 175 successful requests, and should attempt the failed record plus the 160 untouched records. -- PI[gpt-5.6-terra]
