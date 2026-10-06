# Cache mean difference

Author: PI/OpenAI. User request: "ok add it to main please inline with other ones. smoke test it. anothjer agent will run it".

Implementation: `src/steering_lite/variants/cache_mean_diff.py`; public config `CacheMeanDiffC`; benchmark name `cache_mean_diff`.
Source: Belitsky et al., https://arxiv.org/abs/2507.08799, https://github.com/MaxBelitsky/cache-steering/blob/main/src/steering/cache_steering.py.

## Method and deliberate difference from the reference

Extract raw final-token value-cache mean differences from the same positive/negative prompts used by other methods. After the complete prompt forward, add dose times the contrast to that token's cached value, once per selected full-attention layer. Keys and incoming generated-token values remain unchanged.

The extraction pairs are this repository's existing persona-plus-suffix pairs, not the paper's CoT demonstrations. Extraction uses their final real token, which need not be the same token identity or logical position as the generation marker. Behavioural evaluation remains necessary.

This implementation adds no offset token. The first generated token uses bare prefill logits; the edit affects the second token onward. The paper's reference appends an offset token and edits before processing that token. This is an explicit adaptation, not an exact replication of the paper's generation protocol. It preserves the existing benchmark's identical prompt and bare-answer baseline.

Calibration uses an explicit cached prompt/continuation split. Re-prefilling the whole prompt plus generated answer would edit the answer's final cache entry after all scored logits were computed and falsely report zero KL.

## Smoke checks

- Existing pipeline: registration, real extraction, calibration, nonzero continuation-logit change, save/load below 1e-4.
- Real tiny-Llama cache: left/right padded batches, final real token only, exact value-edit formula at zero and both coefficient signs, keys unchanged, prompt logits unchanged, cached continuation changed, no repeat edit on later decode or after detach.
- Cached continuation scoring agrees with token-by-token decoding; zero coefficient has near-zero KL, nonzero dose has positive KL, first-token KL stays near zero.
- Tiny Qwen3.5 hybrid: select a full-attention layer, generate through surrounding recurrent layers, check scorer argmax against greedy generation, and check the zero-dose KL floor.
- Supplied ordinary prefix: preserve the supplied cache, process the rest of the prompt, then edit only the complete prompt's final value.
- BS-bench: `just smoke-bsbench cache_mean_diff`, two rungs on CPU.

Predictions: a residual-style repeated edit changes new values during decode; the cache-history check should reject it. A single teacher-forced prefill scorer reports zero KL despite a cache edit; the cached continuation check should reject it. A right-padding indexing error edits a pad token; the explicit last-real-token formula should reject it.

## Integration and handover

Worktree: `/workspace/2026/lite/steering-lite-cachemd`, branch `feat/cache-mean-diff`, based on `8c6b621` of `dev/prompt-gains-random-reference`.
Literal `main` is `cfe0706`, an older ancestor without this branch's cache infrastructure or BS-bench. Asked the user which integration target they intend; do not merge the full research branch implicitly.

No sweep or judge run is authorized here. A future runner can select `cache_mean_diff` through the existing walk/Modal CLI. Default benchmark layers are all full-attention layers except layer zero; explicit layers are checked at extraction. `--positions user` and the prefill-only sign probe are rejected because neither matches this intervention.

## Results

`library-smoke.log`: "68 passed in 15.93s" (final normal `just smoke`). Initial verification is retained in `library-smoke-initial.log`.

With `BEARTYPE=1`, smoke found three shared interface issues: `ValueGram.install` annotated its reusable cache hook with the method-specific config instead of `SteeringConfig`; `attach` returned lease-wrapped handles but annotated only torch handles; JSON layer lists were not restored to tuples. Fixes keep runtime checks enabled. Original failure logs are retained here.

`targeted-smoke.log`: "24 passed, 44 deselected, 2 warnings in 6.52s" with `BEARTYPE=1`, including the new method and existing value-cache methods.

`library-typed-smoke.log`: "12 failed, 55 passed, 2 warnings in 19.30s". Failures are in untouched pre-existing annotations: SSpace's shared extraction rejects its ablation/scaling config classes; spherical, directional_ablation, chars, and angular_steering annotate one fewer dictionary level than their actual return values. `git diff 8c6b621 --` on those seven variant files is empty. These are recorded, not repaired as part of this method addition.

`bsbench-smoke.log`: "SMOKE_PASS method=cache_mean_diff rungs=2". Two doses on each sign, 20 questions per dose (80 newly generated answers). Vector and calibration were reused from the earlier local smoke; answer files were moved aside so final answers were generated again. Calibration originally reached 1.010 RMS KL on +C; its per-token profile starts with exactly zero KL then nonzero KL from token two, as intended (`bsbench-smoke-prefill-check-failure.log`, calibration completed before the old prefill-only check failed).

The tiny random model gives nonsense, including bare answers (20/20 unfinished). This is runtime evidence, not a behaviour or coherence result.

Commits are separate for review/cherry-pick:
- `e2c217f`: shared runtime type contracts.
- `d66a487`: method, calibration, registration, and initial smoke checks.
- `9143333`: benchmark preflight scores cached continuation instead of prefill logits.
- `454b641`: supplied-prefix timing fix and hybrid scorer/generator checks.

No push or merge into literal main. To bring the code into this benchmark branch without the audit files: `git cherry-pick e2c217f d66a487 9143333 454b641`. Do not use this command on old literal main without integrating prerequisites.

## Independent review and resolutions

`independent-review.md` preserves reviewer-anthropic's review snapshot before final fixes, not a final sign-off. Its benchmark prefill-check blocker reproduced and was fixed in `9143333`; final BS-bench smoke reaches `SMOKE_PASS`. Its missing-proof blocker is addressed by the final normal and typed cache logs above; the full typed suite still has the unrelated annotation failures listed above.

The reviewer also found that a supplied populated prefix was edited before processing the rest of the prompt. `454b641` removes that early edit; the real-model supplied-prefix smoke verifies only the complete prompt's final token changes. It also adds a 2-D mask assertion and hybrid scorer/generator and zero-dose KL checks.

Remaining research limitations: first answer token is unsteered; behavioural sign has not been judged; extraction token identity need not match the generation marker; no GPU timing or full behavioural benchmark was run. The reviewer described the core extraction, padding, cache lifetime, and continuation scorer as consistent by source inspection. These smoke checks establish runtime behaviour only, not steering quality.
