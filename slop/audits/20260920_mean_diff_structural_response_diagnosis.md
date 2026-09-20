# Mean-difference structural response diagnosis

## Observation

[provider evidence](../../outputs/bsbench-v2/provider-evidence/410fc26bb21cc9a70574999105f93df3b424acb7afb12103cecaa0f18bed0e95-000044-failed.json) records:

> `"exception_type": "KeyError"`

The old adapter accessed `response_body["choices"][0]["message"]["content"]` directly. It saved neither the parsed response object nor a response-field path.

## Diagnosis

The evidence cannot distinguish a provider response with no `choices` field from an adapter extracting the wrong field or response shape. It does not support a cause-specific retry. The `$0.0017496` reservation is conservatively recorded at its upper bound in [receipt](../verification/20260920T141130Z_corrected-mean-diff-structural-upper-receipt.json).

## Local change

The adapter now raises `OpenRouterResponseParseError` for absent assistant content and saves a sanitized structural response through the audited callback. It preserves strict schema validation; it does not accept the response or change model, endpoint, or payload.

## Decision

Do not issue another paid mean-difference request on this evidence alone. The remaining work is offline verification and a review decision on how to obtain discriminating provider evidence.

-- PI[gpt-5.6-terra]
