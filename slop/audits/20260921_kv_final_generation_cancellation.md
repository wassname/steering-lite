# KV-cache-Gram final-generation cancellation audit

Target: the interrupted KV-cache-Gram final-generation call started by `slop/verification/20260921T112500Z_kv-cache-gram-final-evaluation.log`.

- provenance: branch `rewrite/bsbench-vjp`; Modal app `ap-tbopwa8x7ZjPo2oyiJw6gm`; function call `fc-01M31VEP8X578MZKGAPS6P7DTV`; container `ta-01M31VEPE7CC2N6XPSPHZQC7NR`.
- local ledger reservation: `2b9e4297a8f5275dbf71f8c6443b1d1523bec12c3f93027b0cdea9145fc8d704`, upper `$0.8843460000000001`.

-- PI[gpt-5.6-terra]

| stage | expected | observed | expected? | consequence |
|---|---|---|---|---|
| cache recovery | reuse KV candidates and candidate judgments | all prior stages are cache hits, then `cache miss final-generation f05418b39b6a` | yes | only final generation was dispatched |
| remote final stage | return a serializable result and create the matching local cache item | Modal loaded Qwen, measured the target and began transfer calibration | partial | remote work started but no serialized result returned |
| cancellation | no cancellation | Modal reports `Received a cancellation signal while processing input` and `Successfully canceled input` | no | stage is incomplete |
| cache persistence | final-generation cache record for the miss identity | no `f05418…` cache record exists | no | no result can be reused or rendered |
| Modal lifecycle | no active remote work before retry | app is `stopped`, `tasks: 0` | yes | retry cannot overlap the old call |
| ledger | every paid reservation reconciled | reservation is unpaired and no Modal usage invoice was recovered | no | record full-upper conservative estimate before retry |

## Primary evidence

The local run reached only the missing final stage:

> `2026-09-21 19:26:06.973 | INFO ... cache reuse compatible candidate-blind edbd4d72fd8c`
>
> `2026-09-21 19:26:06.988 | INFO ... cache miss final-generation f05418b39b6a`

Source: `slop/verification/20260921T112500Z_kv-cache-gram-final-evaluation.log`, local runner log.

Modal confirms that the remote function was not a zero-task lifecycle. It had completed substantial setup and calibration before cancellation:

> `2026-09-21 19:26:28+08:00 ... measure_kl: 100%|██████████| 4/4 [00:04<00:00,  1.10s/it]`
>
> `2026-09-21 19:31:04+08:00 ... calibrate kv_cache_gram: 5it [04:35, 55.16s/it, c=+2.263, kl=0.265, tgt=4.09]`
>
> `2026-09-21 19:32:56+08:00 ... [modal-client] ... Received a cancellation signal while processing input ('in-01M31VEP91RQVSZQFTT84BKPFY:1789989968161-0',)`
>
> `2026-09-21 19:32:56+08:00 ... [modal-client] ... Successfully canceled input ('in-01M31VEP91RQVSZQFTT84BKPFY:1789989968161-0',)`
>
> `2026-09-21 19:34:25+08:00 Stopping app - local client disconnected. Use \`modal run --detach\` to keep apps running even if your local client disconnects.`

Source: `slop/verification/20260921T114100Z_kv-cache-gram-interrupted-function-call.log`, Modal’s persisted function-call log.

## Decision

The call did not complete remotely: cancellation occurred during the final-stage input, no result returned to the local runner, and the required cache record is absent. The function did execute GPU work, so the charge is unknown rather than zero. The ledger will mark this one reservation unresolved then conservatively estimated at its full `$0.8843460000000001` upper bound. After a fresh cache-aware dry preflight, retry only the KV final stage by rerunning the KV condition; its cached candidate work avoids restarting completed work. VJP stages remain stopped until KV completes.
