# VJP-delta final-judge third recovery: upstream HTTP 429

- worker: PI[gpt-5.6-terra]
- failed retry: `slop/verification/20260921T180100Z_vjp-delta-final-judge-third-retry.log`
- code revision: `c6e9c0f9f9cbd4441032766f32d50a05ef1810ab`

## Observation

The third retry reused 258 persisted request caches, retried the earlier zero-cost 429 payload successfully, then persisted three further requests before a new HTTP 429. It logged five cache misses: four successful provider responses and one 429.

The new provider response says:

> `"limit_source": "upstream_provider_shared_pool",`
>
> `"provider_name": "StreamLake",`
>
> `"status": 429`

Source: `outputs/bsbench-v2/provider-evidence/d56fd0a76b4a566e69315b231ad91b6ec1ed5bea8e8e6b8e48d58572777be55e-000005-failed.json`.

The response has no generated content or usage, so reservation `306fa5…` was settled at `$0.00`. The ledger then has `active_unresolved=[]` and `overage=[]`.

Sources: `slop/verification/20260921T180300Z_vjp-delta-final-judge-third-429-zero-receipt.json`; `slop/verification/20260921T180300Z_vjp-delta-final-judge-third-429-zero-settlement.log`; `outputs/bsbench-v2/costs.jsonl`.

The exact current boundary is rebuilt from the cached VJP final-generation plan and production cache identity, rather than inferred from log tails: 262 cached successful responses, two settled failed requests without response caches (one 504, one current 429), and 72 untouched records. This partitions all 336 planned final judgment records.

Source: `slop/verification/20260921T180500Z_vjp-delta-final-judge-third-boundary.json`.

The conservative dry upper is `$29.7911210987 < $50`.

Source: `slop/verification/20260921T180400Z_vjp-delta-final-judge-third-429-retry-dry-preflight.log`.

## Decision

The current failure is another upstream shared-pool rate limit, not a code, data, or strict-schema failure. Keep VJP-cache stopped. After a one-hour bounded backoff, recheck the exact cache boundary, ledger, source revision and budget before retrying only VJP-delta unchanged. -- PI[gpt-5.6-terra]
