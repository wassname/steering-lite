# VJP-delta final-judge fourth recovery: repeated upstream HTTP 429

- worker: PI[gpt-5.6-terra]
- failed retry: `slop/verification/20260921T191100Z_vjp-delta-final-judge-fourth-retry.log`
- code revision: `51c16b8f9ae44530956a2f506f6287ee7db0a9d2`

## Observation

After a one-hour backoff, the VJP-delta retry reused 262 cached final judgment responses and immediately retried the prior pending request. That request returned HTTP 429 in 1.592 seconds, with the same DeepInfra/StreamLake shared-pool limit evidence as the preceding failure. It did not make a new successful request.

> `"limit_source": "upstream_provider_shared_pool",`
>
> `"provider_name": "StreamLake",`
>
> `"payload_sha256": "d56fd0a..."`

Source: `outputs/bsbench-v2/provider-evidence/d56fd0a76b4a566e69315b231ad91b6ec1ed5bea8e8e6b8e48d58572777be55e-000001-failed.json`.

The response has no content or usage. Reservation `44e362…` was settled at `$0.00`; `active_unresolved=[]` and `overage=[]` after receipt import.

Sources: `slop/verification/20260921T191200Z_vjp-delta-final-judge-fourth-429-zero-receipt.json`; `slop/verification/20260921T191200Z_vjp-delta-final-judge-fourth-429-zero-settlement.log`; `outputs/bsbench-v2/costs.jsonl`.

The VJP final judgment cache boundary is unchanged: 262 cached responses, one upper-settled 504, the still-pending d56fd0a request with two zero-cost 429 attempts, and 72 untouched records.

Source: `slop/verification/20260921T180500Z_vjp-delta-final-judge-third-boundary.json`.

## Decision

The one-hour cadence did not clear the shared provider pool. The parent directed a four-hour bounded backoff before the next unchanged VJP-delta-only retry. VJP-cache remains stopped. -- PI[gpt-5.6-terra]
