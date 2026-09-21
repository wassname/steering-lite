# KV final-judge HTTP 429 recovery audit

- worker: PI[gpt-5.6-terra]
- run: `slop/verification/20260921T140500Z_kv-final-judge-third-retry.log` (read in full)
- code revision: `dafeb7193d951790206b50b5a45692c79b9618f0`

| stage | expected | observed | expected? | consequence |
|---|---|---|---|---|
| prior KV work | reuse cached generation and judgments | all earlier stages and 292 judge records hit cache | yes | no GPU work |
| prior timeout | retry the affected 4e8bc1 record | provider response succeeded in 2.366 s and cached | yes | 293 persisted judge records |
| next unfinished record | judge blind AB request | HTTP 429 after 1.196 s | no | stop the KV recovery |
| accounting | resolve provider outcome | 429 receipt settled reservation `6e5820…` at $0.00 | yes | `active_unresolved=[]`, `overage=[]` |

The provider response identifies a shared upstream limit, rather than a response-schema error:

> `"status": 429,`
>
> `"limit_source": "upstream_provider_shared_pool",`
>
> `"raw": "deepseek/deepseek-chat is temporarily rate-limited upstream. Please retry shortly..."`

Source: `outputs/bsbench-v2/provider-evidence/65dde782b7178ee7f422a7430f3326192679679b1a8b99f14f2a4ee64d1fb8ff-000002-failed.json`, generated at the OpenRouter HTTP boundary.

The record immediately before the 429 was successful:

> `"payload_sha256": "4e8bc1b07f6ff5e616f7f8c46da07a33ffa9e6ae223bf2f4b038fb2fef3cf4b6",`
>
> `"elapsed_seconds": 2.3663319009938277,`
>
> `"outcome": "success"`

Source: `outputs/bsbench-v2/provider-evidence/4e8bc1b07f6ff5e616f7f8c46da07a33ffa9e6ae223bf2f4b038fb2fef3cf4b6-000001-success.json`.

The exact reconstruction reports 336 planned request records, 293 persisted successes, one terminal 429 with no cached response, and 42 untouched records. Thus 43 KV cache misses remain, with 22 blind and 21 aware records.

Source: `slop/verification/20260921T141100Z_kv-final-judge-post429-boundary-verified.json`.

## Decision

The KV final-judgment stage is incomplete. This response is a provider shared-pool rate limit; it does not establish a code, data, or strict-schema failure. The 429 response carries no generated content or usage, so the receipt records $0.00. After the parent-authorized one-hour backoff, recheck the ledger, exact cache boundary, unchanged source, smoke evidence, and fresh <$50 upper before retrying KV unchanged. VJP remains stopped. -- PI[gpt-5.6-terra]
