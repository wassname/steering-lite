# KV-cache Gram steering

> wassname: "yes!" to adding the Gram/contrastive value-cache variant in steering-lite, without VJP extraction.

- [x] Add `kv_cache_gram` on `feat/kv-cache-gram`, based on local `origin/main` d260801.
  - Fit per-KV-head uncentered Gram matrices and class means from actual cached values, excluding padding; equal weight per prompt and class.
  - Project the positive-minus-negative mean into the leading rank-r Gram eigenspace, normalize per head. No optimizer or backward pass.
  - Apply `V += C * abs(V @ c) * c` to actual cache values once per entry, before attention reads them. Keep keys unchanged at the intervention site.
  - Support full-attention DynamicCache only. Reject incompatible caches and disabled caching explicitly.
  - Existing prefix entries can be edited; newly appended entries are edited once. Cache mutations persist after detach; reuse across steering contexts is rejected.
  - failure modes: output-hook masquerades as cache steering; repeated historical edits compound; padding or prompt length drives extraction.
  - deliverable: variant, existing API/save-load/calibration integration, selectable benchmark method.
- [x] Verify through the existing real-model functional pipeline on CPU.
  - success: actual cached V changes, same-layer K does not, later decode changes after hooks detach.
  - likely failure: hook does not intercept cache reads; nonzero cache/logit checks detect it.
  - sneaky failure: old values edited again or full-pass scoring differs from cached decode; compare history tensors and teacher-forced logits.
  - Check coefficient zero, sign reversal, label swap, chunk/batch invariance, GQA, save/load and hook cleanup after errors.
  - Inspect failures, repair implementation, and rerun before interpreting behaviour.
  - deliverable: saved test log plus independent review; no leaderboard score or coherence claim from a random model.

## Evidence

- `slop/verification/20260918_post-review-tests.log`: `46 passed in 8.51s` after review fixes.
- Tests cover exact value-edit math, unchanged same-layer keys, coefficient zero/sign symmetry, label swap, fit batch invariance, ordinary-prefix promotion, no repeated history edit, detach/reactivation, attached save/load dtype identity, GQA shapes, and tiny Qwen3.5 hybrid generation.
- `slop/verification/20260918_optional-transformers-import.log`: core package import succeeds when the optional Transformers import is blocked.
- Independent review: OpenAI reviewer found five lifecycle/dependency risks. The implementation now handles empty hybrid caches, optional Transformers import, same-vector reattachment, cache deactivation on detach, and runtime use of serialized buffers.
- No behavioural effectiveness claim yet. The leaderboard job remains the discriminator.

## UAT
`sl.KVCacheGramC(layers=(...), r=16)` fits and runs through `Vector` and the existing sweep CLI. Fit only once; inference uses the stored direction. This first version edits both prefill and newly appended values, not only the original prefix. Published leaderboard rows remain unchanged until measured on comparable runs.

## Method limits
Gram accumulation gives the right singular-vector basis in exact arithmetic, not all SVD factors; it squares the condition number. Float64 statistics reduce roundoff. The leading basis represents energy, not proven semantics. Whole-prompt contrasts can encode instruction wording. Positive/negative projection signs are relative to the extracted axis, not concept-presence labels.

-- PI/OpenAI
