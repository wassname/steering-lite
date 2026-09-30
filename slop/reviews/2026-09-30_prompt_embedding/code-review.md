# Follow-up review: prompt embedding sweep prototype (pre-GPU)

Read: `src/steering_lite/prompting.py`; `walk.py` `generate`/`generate_batch`/`check_prompt_embeddings`/`prompt_sweep`/`walk_done`/`walk` routing; `results.py` `fixed_grid`/intro/`COLORS`/`LABELS`; `web/src/main.jsx:160`; `data.py` seeds; `run_modal.py` `cached_on_volume`; both smoke logs.

## Observed (from logs, not inferred)

- `prompt-embedding-smoke.log`: `PROMPT_SCALE_CHECK_PASS model=Qwen3ForCausalLM C1_logits=exact C1_greedy_ids=exact outside_mask=unchanged decode=unscaled C4_max_logit_delta=10.7 mask_tokens=[10,10,10]`; `PROMPT_C1_CACHE_PASS` for both sides against the 20 cached `prompting_s0/*_C1.jsonl`; `SMOKE_PASS … fixed_grid=True boundary_confirmed=False`.
- `prompt-engineered-embedding-smoke.log` ends after the bare health line: **no evidence yet for the engineered instruction** (longer span, apostrophes, `-C` text differs from `+C`; only `+C` goes through `check_prompt_embeddings`, though `instruction_mask` asserts decode round-trip per batch for both sides).

## Previous blockers — status

1. Context-manager fit: resolved (`generate` builds per-batch mask/embeds; `cached_answers` passes `scaled_instruction`).
2. Prefix slicing: resolved (`generate_batch` asserts `output[:, :n] == input_ids`).
3. Results endpoint: resolved (`sweep_kind`, `boundary_confirmed: False`, `stop_reason: fixed_grid`, marker text "last admissible tested dose", intro note, jsx warning; `calibration.py` skips walks without `c0`; `REPORT_SEEDS["dev"]=range(1)` includes seed 0).
4. Qwen3.5 code path: deferred to the in-job preflight, which fails fast before any sweep answers. Acceptable.

## Remaining correctness items

**Should fix before paying (cheap, avoids a false-negative preflight):**
- `check_prompt_embeddings` asserts `torch.equal(reference, logits)` between two separate forwards. On GPU bf16 with sdpa/FLA kernels, a spurious mismatch from kernel nondeterminism would kill the job and be indistinguishable from a real bug. Add `assert torch.equal(reference, model(**batch, use_cache=False).logits)` (reference vs itself) *before* the embeds comparison, so a failure names the cause. If nondeterminism is found, compare greedy ids / `allclose` and record it.

**Note, not a blocker:**
- The scaled span is 10 tokens and the log shows it decodes to `"Answer as someone who is sycophantic.\n\n"`: the `.\n\n` merge means the paragraph separator is scaled/zeroed along with the instruction. The `.strip()` in `instruction_mask` hides this. Consistent across batch and sides, but the certificate/report should say "instruction plus its trailing separator token", not "only the persona instruction".
- `prepare_inputs_for_generation(..., is_first_iteration=, next_sequence_length=)` are transformers-5.x generic kwargs; a model class that overrides this method with a different signature raises `TypeError` in preflight (fail fast, fine, but the error will look like a bug in the check rather than a model incompatibility).
- `walk_done` for sweeps requires `prompt_gains == list(PROMPT_GAINS)`; any future grid change forces a full recompute of the certificate but answers stay cached per gain. Intended.
- `results.py` `frontier()` docstring still says "last coherent dose"; the plot label is corrected. Cosmetic.

## Missing evidence

Engineered-prompt smoke completion (`PROMPT_SCALE_CHECK_PASS` + both `PROMPT_C1_CACHE_PASS` lines for `prompting_engineered_scale`). Do not launch the engineered dev run until that log shows them.