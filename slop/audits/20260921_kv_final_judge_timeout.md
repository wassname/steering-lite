# KV-cache-Gram final-judge timeout audit

Target: the retry after KV final generation was cached, ending in the OpenRouter read timeout.

- provenance: `slop/verification/20260921T114500Z_kv-cache-gram-final-retry.log` (294 lines, read in full); branch `rewrite/bsbench-vjp`.
- affected request: payload `cde04006…`, reservation `18d7a16e…`, DeepSeek judge through OpenRouter; strict `demo_rating` schema.

-- PI[gpt-5.6-terra]

| stage | expected | observed | expected? | consequence |
|---|---|---|---|---|
| KV candidate stages | reuse completed work | all candidate cache stages are compatible hits | yes | no candidate GPU or judge retry |
| final generation | 84 numbered generation records plus transfer predictions | cache `f05418…` persisted with 84 answers and 84 health records | yes | GPU stage is complete and must not be rerun |
| final judgments | persist every judged response | 178 preceding requests cached; request 179 waited 301.501s then timed out | no | only one judge cache key is missing |
| provider accounting | reconcile every request | provider response/bill is unknown for the failed payload | no | conservatively estimate its `$0.0022644` reservation upper |
| Modal lifecycle | no remote Modal work before retry | retry app `ap-odCDvNLeX3XC9ahOeKvFlA` is stopped with zero tasks | yes | retry has no GPU task |

## Primary evidence

The run reused every earlier KV stage, performed the missing final generation once, then began final judgments:

> `2026-09-21 19:46:21.670 | INFO ... cache reuse compatible candidate-blind edbd4d72fd8c`
>
> `2026-09-21 19:46:21.672 | INFO ... cache miss final-generation f05418b39b6a`
>
> `2026-09-21 20:02:53.528 | INFO ... cache miss final-judgments 77e2c9351c99`

Source: `slop/verification/20260921T114500Z_kv-cache-gram-final-retry.log`, the actual retry log.

The persisted final-generation record has 84 planned responses and 84 health records; its remote cost receipt reports 969.315 seconds. It establishes completed local persistence, not a provider invoice.

The failed request evidence says:

> `"dispatch_started_at": "2026-09-21T12:35:22.105000+00:00",`
>
> `"elapsed_seconds": 301.5014542990248,`
>
> `"exception_type": "TimeoutError",`
>
> `"outcome": "failed",`
>
> `"response_schema": "demo_rating"`

Source: `outputs/bsbench-v2/provider-evidence/cde04006c77c3e6c33bfae0488f2ae8adeed147a0eac62cd7b1d5491baeed83e-000179-failed.json`, generated at the callback boundary.

## Decision

This is an isolated final-judge request failure after successful cached generation and 178 successful new judge responses. The payload did not become a cache record, so rerunning the KV condition will reuse all completed work and issue only this payload. The account metadata request returned HTTP 200 for both `key` and `credits`, but it cannot attribute this one timed-out request. Record its full reservation upper, wait the parent-authorized 15 minutes, then retry unchanged: DeepSeek model, payload, strict schema, 10-second pacing and 300-second read limit. VJP remains stopped. If that same payload fails again, stop after reconciliation and report its new evidence before another paid request.
