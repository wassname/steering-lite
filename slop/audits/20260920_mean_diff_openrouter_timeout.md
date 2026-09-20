# Mean-difference candidate judging: OpenRouter no-response timeout

- scope: approved canonical real sweep, `mean_diff` candidate judging only.
- decision: do not change model, endpoint, payload, question set, score, or 10-second pacing. The unresolved blind request is not reconciled because no response or provider request ID exists.

## Observations

### Production sweep log — local client observation

- source: `slop/verification/20260920T030842Z_canonical-full-after-random-terminal.log`.
- epistemic context: production command stdout/stderr. It can establish local cache order and the client exception, but cannot establish whether OpenRouter completed billing after the socket stopped waiting.

> 2026-09-20 11:08:47.438 | INFO ... cache hit calibration-candidates 5838cba24e92
> 2026-09-20 11:08:47.468 | INFO ... cache hit candidate-judgments 75e031822287
> 2026-09-20 11:08:47.514 | INFO ... cache miss calibration-candidates cbacb6c66969
> 2026-09-20 11:11:25.209 | INFO ... cache miss candidate-judgments a3ffc00bfb87

This shows bare, prompting, and random were reused; the new work began at mean-difference candidate calibration.

> File "/home/code/.local/share/uv/python/cpython-3.13.12-linux-x86_64-gnu/lib/python3.13/urllib/request.py", line 1323, in do_open
>     r = h.getresponse()
> ...
> TimeoutError: The read operation timed out
> error: recipe `sweep` failed on line 16 with exit code 1

The timeout occurred while reading an HTTPS response; it was not an HTTP 429 or a completed response with omitted cost.

### Reconstructed request and ledger/cache reconciliation

- source: `slop/verification/20260920T032644Z_mean-diff-timeout-request-reconstruction.log` and `slop/verification/20260920T032741Z_mean-diff-timeout-cache-ledger-audit-repair.log`.
- epistemic context: production request-builder replay against the cached mean-difference candidates. It makes no network call and stops before judge execution.

> {"blind": true, "comparison_id": "f1f9c8b1d8eb114b", "content_chars": 1675, "input_tokens_upper": 2000, "max_tokens": 1200, "model": "deepseek/deepseek-chat", "order": "AB", "output_tokens_upper": 1200, "payload_sha256": "250b555b067df97f5680d0eb92204e79a779a9b09e655b6eb144eb46b64a309c", "question_id": "BSV2-002", "question_number": 2, "request_key": "0313a31ea7d5fcdc92d089cab9eefe47475de29f4119b2a0ad76e73acb204301", "response_schema": "blind_change_description", "schema": "bsbench-judge-request-v1", "side": null, "temperature": 0}

This is the exact affected payload identity. It contains no prompt text in this audit.

> reconstructed_request_records=96 unique_request_keys=84 aware=48 blind=48
> ledger_reserved_matching=58 ledger_settled_matching=57 ledger_unresolved_matching=1
> cache_response_matching=57
> unresolved_keys=['0313a31ea7d5fcdc92d089cab9eefe47475de29f4119b2a0ad76e73acb204301']
> settled_actual_usd=0.021552308

The lower unique-key count is expected: identical candidate answer pairs share a cached request. Every dispatched preceding unique request has both a ledger settlement and cache response. The 26 unreserved keys were never reached after the failure. The candidate-judgments stage has no partial cache record because it writes only after all reconstructed requests return.

The timed-out reservation is `d4cd25efb75817c4fd95ac0151282f26a13b0b9fc4677e58e9b3de8eb8512f95`, upper `$0.0017496`, with ledger reason `judge_request_failure`. It was subsequently settled at that full upper via `slop/verification/20260920_mean-diff-timeout-upper-receipt.json`: `unknown provider outcome; charged at upper`. This is append-only and does not claim a known provider bill. Mean-difference calibration itself is preserved and conservatively estimated at its `$0.884346` stage upper: reservation `25156108d6fb47e51b8b80f2802e669246866600caf8b3dd98990c4384c6ccda`, provider pending usage `148.080115213` seconds and 200 persona pairs.

### Non-generation key metadata — provider account observation

- sources: `outputs/bsbench-v2/provider-evidence/openrouter-metadata-20260920T015551Z.json` and `outputs/bsbench-v2/provider-evidence/openrouter-metadata-20260920T032823Z.json`.
- epistemic context: OpenRouter `/api/v1/key` and `/api/v1/credits` account summaries. They report aggregate key usage, not request-level attribution.

> "usage": 0.033085887,
> "limit_remaining": 49.966914113,
> "total_usage": 4661.801028884

> "usage": 0.099578293,
> "limit_remaining": 49.900421707,
> "total_usage": 4663.438449544

The key-usage delta is `$0.066492406`. After the earlier metadata timestamp, the ledger has 193 settled judge reservations totalling `$0.0679039233` and the one unresolved upper `$0.0017496`; the aggregate key delta is `$0.0014115173` *less* than settled ledger cost. Therefore there is no positive unexplained key-use delta that can support charging the failed request, but account metadata cannot prove that it was free. Keep its upper unresolved.

## Root cause and recovery change

The prior transport used `urlopen(..., timeout=90)`. The affected request was reserved at `03:23:28.446Z`, while the process ended after the exception at `03:25:18Z`: about 110 seconds of observed wall time, despite the 90-second socket-read setting. This is evidence that 90 seconds was below this request's observed end-to-end wait, but not provider documentation that 180 seconds is sufficient. OpenRouter metadata supplies no supported latency bound; its only rate-limit field says it is deprecated.

The script now applies a fixed 180-second read timeout only around OpenRouter judge calls, retaining the existing 10-second start-to-start pacing. It does not touch `src/`, so prior paid cache identities remain valid. The audit wrapper now persists `bsbench-openrouter-no-response-v1` evidence for `TimeoutError`, `URLError`, `RemoteDisconnected`, or `ConnectionResetError`: endpoint, payload hash, model, response schema, exception class, and monotonic elapsed seconds only. It does not store API keys or request content.

Regression evidence: focused tests `16 passed` and full offline tests `138 passed` in `slop/verification/20260920T033058Z_mean-diff-timeout-audit-focused-tests.log` and `slop/verification/20260920T033140Z_mean-diff-timeout-audit-full-tests.log`.

## Retry plan

One retry is proposed, not run: the identical canonical `mean_diff` recovery command. Existing 57 judge responses and mean-difference calibration are cache hits; the same unresolved request key is first and retains the same endpoint, model, serialized payload, temperature, token uppers, and 10-second pacing. It uses only the 180-second read timeout. No later requests/methods may dispatch before that request settles. If it fails or has unknown outcome again, stop immediately and retain its new sanitized no-response evidence.

After the append-only upper settlement, the unresolved set is empty and the aggregate dry preflight remains `$44.3855922359 < $50`; it includes `$7.4088154359` existing commitments plus expected work, one full retry reserve, and unresolved-work reserve. Evidence: `slop/verification/20260920T033618Z_mean-diff-timeout-post-reconciliation-dry-preflight.log`.

## Epistemic summary

- The client stack trace and ledger agree that one request received no usable response. They do not say whether OpenRouter processed it after the client timeout.
- The cache/ledger pairing is independent enough to establish the 57 completed prior requests, but it shares the local process as the source of the request sequence.
- The aggregate account delta is lower than prior settled costs; it weakly argues against a large additional charge, but is not request-level proof.
- Calibrated take: `p ≈ 0.65–0.80` that this was a transient long/no-response transport event rather than a model or quota failure. The cheapest counterevidence is another identical request with 180 seconds that fails similarly.

-- PI/OpenAI
