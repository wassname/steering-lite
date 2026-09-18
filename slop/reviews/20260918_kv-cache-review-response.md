# KV-cache Gram review response

Reviewer: `reviewer-openai`, read-only review of the uncommitted branch. Anthropic review did not start because Fable returned HTTP 429 `credits_required`.

## Findings and response

1. > "Hybrid-model `generate()` likely crashes before prefill"
   - Fixed empty hybrid-cache promotion before examining linear-attention layer state.
   - Added `test_kv_cache_gram_hybrid_generate` with a four-layer tiny Qwen3.5 hybrid model. It generates two cached tokens through a selected full-attention layer.
2. > "New unconditional import breaks installations without Transformers"
   - Made the cache integration optional at package import. Use now fails with an explicit extra/version message.
   - `slop/verification/20260918_optional-transformers-import.log` blocks the Transformers import and imports `steering_lite` successfully.
3. > "A cache cannot survive reattachment of the same vector"
   - Reattachment now compares coefficient and tensor content, not a transient dictionary identity.
   - Same-vector reattachment is tested. Different state remains rejected.
4. > "Detaching leaves future steering active on returned caches"
   - Each attachment owns a lease. Detach disables future cache edits while preserving the already-edited history.
   - The test captures raw `v_proj` output after detach and checks the appended cached value is unchanged.
5. > "Runtime uses different state from attached-state serialization"
   - Runtime now reads the installed buffers after dtype conversion.
   - Attached-model save/load is tested with bfloat16 buffers on a float32 model.

The reviewer also requested an independent formula check and GQA/hybrid coverage. Both are in `tests/test_pipeline.py`. Final evidence: `slop/verification/20260918_post-review-tests.log` reports `46 passed in 8.51s` after the fixes.

No behavioural-effectiveness claim follows from these software checks.

-- PI/OpenAI
