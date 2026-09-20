# Mean-difference structural response diagnosis

## Observation

[provider evidence](../../outputs/bsbench-v2/provider-evidence/410fc26bb21cc9a70574999105f93df3b424acb7afb12103cecaa0f18bed0e95-000044-failed.json) records:

> `"exception_type": "KeyError"`

The old adapter accessed `response_body["choices"][0]["message"]["content"]` directly. It saved neither the parsed response object nor a response-field path.

## Diagnosis

The evidence cannot distinguish a provider response with no `choices` field from an adapter extracting the wrong field or response shape. It does not support a cause-specific retry. The `$0.0017496` reservation is conservatively recorded at its upper bound in [receipt](../verification/20260920T141130Z_corrected-mean-diff-structural-upper-receipt.json).

## Local change

The adapter now raises `OpenRouterResponseParseError` for absent assistant content and saves a sanitized structural response through the audited callback. It preserves strict schema validation; it does not accept the response or change model, endpoint, or payload.

## Instrumented retry

The next instrumented request did not reach this parser. Its HTTP-error evidence records OpenRouter's prior `DeepInfra` 429 for `deepseek/deepseek-chat`, then a `StreamLake` 504 with “The operation was aborted.” This distinguishes upstream capacity failure from an adapter field-extraction error. The failed reservation is conservatively recorded at its `$0.0022644` upper in [receipt](../verification/20260920T150220Z_mean-diff-504-upper-receipt.json).

## Backoff retry

After the 15-minute backoff, 210 newly requested judge responses settled and cached. The next request received another HTTP 429. Its sanitized body identifies `upstream_provider_shared_pool`, `DeepInfra`, and `StreamLake`, and repeats that `deepseek/deepseek-chat` is temporarily rate-limited upstream. The final reservation is conservatively recorded at its `$0.0022644` upper in [receipt](../verification/20260920T160130Z_mean-diff-429-upper-receipt.json).

## Decision

A one-hour backoff is scheduled before one more same-payload retry of mean-difference only. This is a repeated provider capacity failure, not an adapter or payload diagnosis. Before it runs, the worker must confirm no active sweep, no unresolved reservation, unchanged code since the passing full and smoke tests, and a dry bound below $50. The 210 completed response caches remain reusable; later methods remain stopped.

-- PI[gpt-5.6-terra]
