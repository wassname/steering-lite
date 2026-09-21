# VJP-delta final-judge second recovery: upstream HTTP 429

- worker: PI[gpt-5.6-terra]
- failed retry: `slop/verification/20260921T163700Z_vjp-delta-final-judge-retry.log`
- code revision: `c3b77e4c972cebbe375b365149eb194d685e7681`

## Observation

The unchanged VJP-delta retry reused its prior 175 cached final judgments. It then persisted 84 more successful requests before the next request returned HTTP 429. The current boundary is 259 cached successes, one earlier upper-settled HTTP 504, one zero-cost HTTP 429, and 76 untouched records, against 336 planned final judgment records.

The provider evidence says the 429 is an upstream shared-pool limit:

> `"limit_source": "upstream_provider_shared_pool",`
>
> `"provider_name": "StreamLake",`
>
> `"raw": "deepseek/deepseek-chat is temporarily rate-limited upstream..."`

Source: `outputs/bsbench-v2/provider-evidence/86e2c371bd4e3beffbfb92f881e13498bb87063c036bd2710310bc59613d22fb-000085-failed.json`.

This response has no generated content or usage. Its reservation `e1306e…` was therefore settled at `$0.00`, leaving `active_unresolved=[]` and `overage=[]`.

Sources: `slop/verification/20260921T165900Z_vjp-delta-final-judge-second-429-zero-receipt.json`; `slop/verification/20260921T165900Z_vjp-delta-final-judge-second-429-zero-settlement.log`; `outputs/bsbench-v2/costs.jsonl`.

The post-settlement conservative dry upper is `$29.7894003064 < $50`.

Source: `slop/verification/20260921T170000Z_vjp-delta-final-judge-second-429-retry-dry-preflight.log`.

## Decision

This is an upstream rate limit, not a code, data, or strict-schema failure. VJP-cache remains stopped. Take a one-hour bounded backoff, then recheck the ledger, source revision, cached boundary, and budget before retrying VJP-delta unchanged. -- PI[gpt-5.6-terra]
