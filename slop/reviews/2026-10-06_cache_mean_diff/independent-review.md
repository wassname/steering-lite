# Independent review: `cache_mean_diff` (values-only cache steering, arXiv 2507.08799)
Reviewer: reviewer-anthropic (read-only). Repo `/workspace/2026/lite/steering-lite-cachemd`, branch `feat/cache-mean-diff`.
Scope read in full: `src/steering_lite/variants/cache_mean_diff.py`, `variants/value_gram.py`, `calibrate.py`, `attach.py`, `vector.py`, `config.py`, `target.py`, `positions.py`, `tests/test_pipeline.py`, `scripts/bsbench/walk.py`, `scripts/bsbench/results.py` (colour/label handling), `AGENTS.md`, `justfile`, `slop/reviews/2026-10-06_cache_mean_diff/*` (diff, results.md, all logs). Also read transformers 5.12.1 `modeling_qwen3_5.py` and `cache_utils.py` in the shared venv to check chunked-continuation and hybrid-cache assumptions.
Legend: **[obs]** = directly observed in files/logs; **[inf]** = my inference from code reading, not executed.
---
## 0. Evidence status (read first)
- **[obs] The supplied `implementation.diff` is stale relative to the working tree.** Present in the tree but absent from the diff: `config.py:58-59` (`from_dict` now coerces `layers` to tuple), `attach.py:95/310` (return hint changed to `list[_Handle]`), `value_gram.py:294` (`install(..., cfg: SteeringConfig, ...)` instead of `ValueGramC`), `walk.py:112-113` (`parser.error` for `--probe/--vjp-check` with cache_mean_diff), `tests/test_pipeline.py:647-705` (two new tests), `README.md:201`. Please regenerate the diff before any merge decision; this review is against the tree as read.
- **[obs] No passing BS-bench smoke log exists.** `bsbench-smoke.log` (the latest, 12:55:59) ends in a traceback at `Vector.load → from_dict` (beartype: `layers=[1, 2]` is a list). `results.md` lists "BS-bench: `just smoke-bsbench cache_mean_diff`, two rungs on CPU" as a smoke check, but no log shows `SMOKE_PASS`.
- **[obs] No passing `BEARTYPE=1` pytest log exists.** `targeted-smoke.log` (BEARTYPE on) shows 5 failures (3 of them for pre-existing methods `value_gram`, `vjp_value`) at `Vector.load`; `library-smoke.log` (66 passed) shows no beartype frames, so it was run without `BEARTYPE=1`. The `config.py` tuple fix was presumably applied after; a rerun log is missing.
---
## 1. Blockers
### B1. `walk.py:619-623` asserts steering changes *prefill* logits; cache_mean_diff cannot by design → `just smoke-bsbench cache_mean_diff` will fail with "steering changed no logits"
**[obs]** `scripts/bsbench/walk.py:619-623`:
```python
encoded = tokenizer(prompts[0], return_tensors="pt", add_special_tokens=False).to(args.device)
with torch.inference_mode():
    base_logits = model(**encoded).logits
    with vector(model, C=GRID[start["+C"]]):
        assert not torch.equal(base_logits, model(**encoded).logits), "steering changed no logits"
```
**[obs]** `cache_mean_diff.py:112-119` edits the cache in a *forward hook on the top-level model*, i.e. after `lm_head` has produced the prefill logits. The method's own test asserts bit-exact equality of prefill logits: `tests/test_pipeline.py:664` `torch.testing.assert_close(prefix.logits, bare.logits, rtol=0, atol=0)`.
**[inf]** Therefore `torch.equal(base_logits, steered_prefill_logits)` is `True` and the assertion fires on every walk (smoke or Modal), right after `calibration_c0`. The latest `bsbench-smoke.log` never reached this line because it crashed earlier at `Vector.load`; once that is fixed this is the next failure.
**Affected inputs:** every `walk.py cache_mean_diff` invocation (smoke and real sweeps). `README.md:201` currently promises `just smoke-bsbench cache_mean_diff` works.
**Disproving check:** run `just smoke-bsbench cache_mean_diff` (free, CPU). If it prints `SMOKE_PASS`, I am wrong.
**Fix sketch (parent's choice):** for `prompt_cache_only` methods compare `REGISTRY[m].score_continuation(model, ids, ids[:, -2:])` (row 1 differs) or compare one greedy `generate` step beyond the first token; or skip with a logged reason, mirroring the `--probe` exclusion.
### B2. Smoke evidence incomplete: parent-claimed smoke set has not been shown green
**[obs]** See §0. The only green artefact is `library-smoke.log` (no BEARTYPE). Given AGENTS.md "`just check` runs both", acceptance needs: (a) `BEARTYPE=1 uv run --extra test --extra hf-test pytest -q tests/test_pipeline.py` green, (b) `just smoke-bsbench cache_mean_diff` ending in `SMOKE_PASS`, (c) `just smoke-bsbench` (default `mean_diff`) still green after the `attach.py`/`config.py` changes.
---
## 2. Correctness review of the algorithm (what I verified by reading)
Intended: `c_l = mean_n V_pos[last] − mean_n V_neg[last]` (raw, per full-attention layer), one-shot `V_prompt[last] += coeff·c_l` after complete prefill, future incoming values untouched; first generated token unsteered; calibration scores the cached continuation.
| Aspect | Verdict | Evidence |
|---|---|---|
| Extraction indexing under right padding | OK **[inf]** | `_encode` pads right (`vjp_resid.py:67-75`); `_last_indices(attention_mask)` gives the last real index; `layer.values[rows, :, last, :]` with two advanced indices separated by a slice → `[b, h, d]` (advanced dims first), `.sum(0)` → `[h, d]`. Causal attention means right pad does not contaminate the last real token. |
| Mean over prompts | OK | per-batch `sum(0)/len(prompts)` accumulated across batches; pos/neg equal counts asserted (`cache_mean_diff.py:80-81`). |
| Stored shape / multi-round | OK | `c.unsqueeze(0)` → `[1,h,d]`; `directions.sum(0)` at edit handles `Vector + Vector`; `Vector * k` scales linearly. |
| Edit location at inference | OK **[inf]** | `edit_prompt` uses the 2-D `attention_mask` kwarg of the top-level call → `s-1` for left-padded batches (walk `generate`, `tokenizer.padding_side="left"`, `walk.py:222`), per-row last index for right padding, `s-1` if no mask (single unpadded prompt). Tested both sides in `test_cache_mean_diff_one_shot_prompt_edit`. |
| No repeat edit on decode / after detach | OK | `edited` flag; `_edit` override is identity so `update()` never touches incoming values; `_CacheHookHandle.remove()` kills the lease so a surviving cache object cannot be re-edited. |
| Same cache reused under a re-attach with different coeff | fail-fast OK | `inject_cache` → `matches()` raises `RuntimeError` (`value_gram.py:311-318`). |
| HF `generate` cache flow | OK **[obs/inf]** | HF's decoder creates `DynamicCache(config)` only if `past_key_values is None` (`modeling_qwen3_5.py:1171`); the decoder pre-hook runs first and injects/promotes. `promote` on an empty hybrid cache returns a fresh `PromptValueCache`; `get_seq_length()` is hybrid-safe (`cache_utils.py:1249-1270`). |
| Calibration continuation scorer | OK **[inf]** | `score_continuation` returns `[prefill_last, logits(gen[:-1] | cache)]` → row *i* predicts `gen[i]`; matches `logp_base = model(full).logits[0, n_p-1 : n_p-1+n_gen]`. Base pass runs detached (outside `with v(model)`), so no hook interference. |
| Chunked continuation on hybrid Qwen3.5 | OK by source **[obs]** | transformers 5.12.1 `modeling_qwen3_5.py:479-487` prepends the cached conv state for multi-token cached continuation and passes `initial_state=recurrent_state` to the chunk kernel, so one-shot scoring is designed to equal streamed decode on linear-attention layers too. Only verified by test on Llama (see C3). |
---
## 3. Non-blocking findings (ordered by importance)
### C1. First-token-unsteered deviation is a real effect-size risk on BS-bench, not just a calibration footnote
**[obs]** BS-bench prompts end with `<|im_start|>assistant\n<think>\n\n</think>\n\n` (bsbench-smoke.log, "generation input 0"), so the first *answer* token is the unsteered one. Stance on the premise-acceptance axis frequently lands in token 1 ("Great…", "Actually…", "There is no…").
**[inf]** Compared with the paper's offset-token protocol, this variant forfeits the single most stance-laden position and may under-perform mean_diff for reasons unrelated to the value-cache mechanism. Deliberate per the parent; I flag it so a weak result is not mis-attributed to the method. Calibration impact is negligible (per-t KL[0] ≡ 0 dilutes `kl_rms` by ~1/T at T=50).
### C2. Sign is uncertified for this method
**[obs]** `--probe` is rejected (`walk.py:112-113`, `607`); the `sign_v2` artefact that other methods get will not exist. `+C` relies on `c = pos − neg` and the persona pairs' orientation. Not a bug; the results pipeline should treat +C/−C labels as unverified for this method.
### C3. Smoke evidence of dose response is very weak
**[obs]** `targeted-smoke.log` calibration table for `cache_mean_diff` shows `mean=rms=p95=max=0.0000` at every c from 0.447 to 7.155 (5 decimals), identical steer/base rollout text; `value_gram` at the same c shows 0.003. The dedicated test only requires `steered.kl_rms > zero.kl_rms + 1e-7` (`tests/test_pipeline.py:703`) and `diff > 1e-6`.
**[inf]** Plausibly just the tiny random model (small value norms, layer 1 of 2), but the smoke does not demonstrate that calibration can ever reach a target. Cheap strengthening below (§4).
### C4. Promoted populated prefix cache edits the *prefix's* last token, not the prompt's last token
**[obs]** `PromptValueCache.promote` (`cache_mean_diff.py:62-67`) calls `edit_prompt()` (mask=None) *before* the new tokens are processed; `finish_prefill` is then a no-op (`edited=True`).
**[inf]** Semantics differ from "final prompt token" whenever a user passes a prefix cache plus new prompt tokens. Not exercised by walk/calibration (both start from an empty cache). Simplest fix if wanted: delete the `promote` override and let `finish_prefill` do the one edit after the forward (it then lands on the true last token and sees the attention mask). Docstring currently only mentions the hybrid limitation.
### C5. `_last_indices` assumes a 2-D mask; no shape assert
**[obs]** `cache_mean_diff.py:33-34`, `116-117`. A 4-D/additive mask passed explicitly would silently yield wrong indices. HF `generate`/walk always pass 2-D. Hardening: `assert mask.ndim == 2`.
### C6. `extract` mutates `cfg.layers` in place when `None`
**[obs]** `cache_mean_diff.py:82-85`. `train()` already called `find_targets(model, cfg)` with `layers=None` (all blocks) before extract; harmless because `extract_from_prompts` ignores `targets`, and attach re-resolves from the now-set tuple. Side effect on the caller's config object; `value_gram` avoids this by computing a local `layers`. Style only.
### C7. Pre-existing beartype latent bugs were surfaced, fixed outside the diff
**[obs]** `from_dict` list→tuple, `attach` return type vs `_CacheHookHandle`, `ValueGram.install` type hint. These affect `value_gram`/`vjp_value` too and are good fixes, but they must be in the reviewed diff and covered by a `BEARTYPE=1` run (see B2).
### C8. Minor perf
`layer.values.clone()` per selected layer once per prefill (`cache_mean_diff.py:53`): ~tens of MB transient per layer at batch 200 on 4B; acceptable, in-place would be fine post-forward but clone is the safer choice. No action needed.
---
## 4. Simplest discriminating smoke checks (all CPU/free, no sweeps)
1. **B1 repro:** `just smoke-bsbench cache_mean_diff` → expect failure at `walk.py:623` ("steering changed no logits"); after fix, expect `SMOKE_PASS … rungs=2`.
2. **Beartype green:** `BEARTYPE=1 uv run --extra test --extra hf-test pytest -q tests/test_pipeline.py` → 66 passed; attach log.
3. **Greedy self-consistency (strongest cheap check of scorer == generator):** on tiny Llama or tiny Qwen, with the vector attached at a large C (e.g. 1e3) and `do_sample=False`, generate T=8 tokens, then assert `score_continuation(model, prompt, gen).argmax(-1) == gen` for all positions (ties aside). This fails if the chunked scorer and the step-wise generator ever diverge (padding, mask, cache lifetime).
4. **Dose response actually visible:** tiny Llama, `measure_kl` at C ∈ {0, 1e2, 1e3}: expect `kl_rms(0) < 1e-6`, monotone increase, and `per_t_p95[0] == 0` at all C (first token unsteered). Addresses C3.
5. **Hybrid floor:** repeat check 4's C=0 case on the `Qwen3_5ForCausalLM` config from `test_cache_hybrid_generate` (layer 2 full-attention): `kl_rms(C=0) < 1e-5` confirms chunked continuation through linear-attention layers has no non-steering KL floor (source says it should not; test proves it).
6. **Left-padded batch == single:** generate the same prompt alone and inside a left-padded batch of 2 at the same C; greedy ids must match for the real prompt. Catches any `_last_indices`/mask-kwarg mismatch under walk's batching.
---
## 5. Summary
- Core algorithm and lifetime handling match the stated intent and are sound by reading; padding (left/right), one-shot semantics, detach, and the calibration scorer are all consistent.
- **Blocking:** `walk.py:623` prefill-logit assertion is incompatible with a post-prefill cache edit, so the benchmark entry point fails; and the claimed smoke set has no passing evidence (bsbench smoke log is a failure; no BEARTYPE=1 green run). The supplied diff does not reflect the tree.
- Residual design risk: unsteered first answer token on BS-bench, uncertified sign.
