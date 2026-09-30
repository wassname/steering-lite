# Design review: prompt embedding gain sweep

Read: `slop/plans/2026-09-30_prompt_embedding_sweep.md`, `scripts/bsbench/walk.py`, `AGENTS.md`, `results.py`/`data.py` (consumer path), `justfile`. Not found in repo: any ml-debug skill file (`find **/*ml-debug*` empty); `../lora-lite/AGENTS.md` is outside scope. No diff or command output was supplied, so everything below is inference from the plan text and current code.

## Blockers (fix before the GPU run)

1. **Context-manager signature does not fit `cached_answers`.** `cached_answers(..., steer)` calls `with steer(): generate(model, tokenizer, prompts, batch_size)` around *all* batches, and `generate` tokenizes internally and calls `model.generate(**batch, ...)`. The plan's `scaled_prompt_embeddings(model, input_ids, mask, C) as embeddings` needs per-batch `input_ids`, so either (a) the prompt-sweep mode gets its own generate that builds embeddings per batch, or (b) the "with" is a forward hook on `model.get_input_embeddings()` that scales output by the mask only on the prefill call (seq len > 1 / `cache_position[0]==0`), which reuses `cached_answers` unchanged. Both are the same intervention; pick one explicitly. With (a) the `with` is decorative (nothing to restore).

2. **Output slicing assumes prompt ids are in `output`.** `generate` returns `output[:, input_ids.shape[1]:]`. In HF decoder-only models, passing `inputs_embeds` **without** `input_ids` yields sequences containing only new tokens, so the slice would silently drop the first `prompt_len` generated tokens (a plausible cause of "all answers empty/truncated"). Passing both is allowed for decoder-only (`_prepare_model_inputs` moves `input_ids` into `model_kwargs`), but assert `output.shape[1] > input_ids.shape[1]` and that `output[:, :prompt_len]` equals the input ids.

3. **Results pipeline treats a new method name as a walk.** `results.py:PROMPTS` only lists `prompting`/`prompting_engineered` as single points; a `prompting-gain` certificate would be plotted as a curve, and `frontier()` marks `curve[-1]` as "× = last coherent dose" (results.py:355, 461). On a finite grid ending at C=16 with no confirmed breakdown, that label is the exact "invented boundary" the plan forbids. Either add the method to `PROMPTS`-style handling or write `post_boundary`/status so the grid end is not presented as a breakdown endpoint. `walk_certificates` (data.py:29) also filters by `REPORT_SEEDS`; confirm seed 0 is included.

4. **Model class / code path unverified.** `Qwen/Qwen3.5-4B` is loaded through `AutoModelForCausalLM` with a hybrid linear-attention stack. Check `type(model).__name__`, whether `prepare_inputs_for_generation` is overridden, and that `inputs_embeds` is accepted at prefill with the hybrid cache. A tiny-random Qwen3 smoke does not cover this.

## Likely implementation bugs

- **Mask via offsets under left padding.** `return_offsets_mapping` needs a fast tokenizer; pad tokens get `(0,0)`. Use strict overlap (`start < span_end and end > span_start`). Qwen BPE may merge `.` with the following `\n\n` into one token, so the last instruction token straddles the boundary; decide and assert `mask.sum(1)` is constant across the batch, and log `tokenizer.decode(ids[mask])` once ("SHOULD equal the instruction").
- **Gain-one identity in bf16.** Multiplying by 1.0 is exact, so C=1 logits should be bit-identical to the `input_ids` path *on the same padded batch*; compare greedy tokens, not just text.
- **C=0.** `input_layernorm(0)=0`, so those positions become near-blank keys/values (Qwen3 has no qkv bias), plus they still carry position. Log this as "blank positions", not "no instruction".
- **Hook variant leaks.** If a hook is used, it must be removed on exit and must fail (not skip) on shape mismatch during decode.
- **Cache path.** `answer_path` uses `f"{side}_C{coefficient:.10g}"`; `C=0.125` → `+C_C0.125.jsonl` is fine, but the method folder must differ from `prompting_s0`, which already holds `+C_C1.jsonl`.

## Assumptions worth stating

Scaling the token embedding only rescales the direct-path residual (`h0 = C·e`); layer inputs are RMS-normalised, so the effect is on the ratio of embedding to layer contributions, consistent with the plan's "less context-sensitive" note. Expect non-monotone curves; a flat curve at C∈[0.5,2] is expected, not a bug.

## Cheapest discriminating tests (tiny model, CPU)

1. `input_ids`-only vs `input_ids+inputs_embeds` at C=1: `torch.equal` on logits and greedy ids, left-padded batch of 3 different lengths.
2. C=4: assert logits differ; assert `embeddings[~mask] == embed(ids)[~mask]` exactly.
3. Second-step decode: hook/embeds path must embed the generated token normally (compare 2-step greedy with `use_cache=True` vs `False`).
4. Assert no parameter changed after context exit (`all(torch.equal(a,b))` over `state_dict`).

## Optional extra experiments (not replacements)

- Matched neutral-prefix control at the same gains (isolates "any scaled prefix" from persona content).
- Log per-rung RMS-KL vs bare using `measure_kl` for comparability with the walk's calibration table, clearly labelled as not iso-KL-calibrated.