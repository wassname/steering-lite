**Missing evidence:** `/tmp/supp/suppressed_activation_subspace.py` does not exist, so faithfulness of `suppressed_mean_diff` to the source repo is **unverified**. Vector `.safetensors/.json` files are not in the worktree; both `modal-dev-suppressed*.log` end at `calib … 0it` with no `WALK_COMPLETE`, so `suppressed_mean_diff` never produced a walk (consistent with its absence from `index.md`).

## Findings

**1. mean_vjp / wiki_mean_vjp are dominated by the residual identity path → they are ≈ "layer‑29 mean‑diff injected at layers 6–24", not a new pullback direction.**
`vjp_delta.py:127-132`: `grad_outputs = c * valid` on `h_target`, sources are block outputs. Since `h_29 = h_l + Σ block_k`, `∂h_29/∂h_l = I + …`, so every valid position's gradient is `c + (attention/MLP pullback)`. `_class_mean_vjp` then averages over positions. In `vjp_delta` the `c` term cancels exactly in `positive - negative` (line 217); in `mean_vjp.py:70-74` it does not. Hence `v_l ≈ unit(c + E[nonlinear])`, and for WikiText the nonlinear term is off‑distribution noise, so `wiki_mean_vjp ≈ unit(c)` at every layer. This matches the table: wiki_mean_vjp (+0.71) ≈ mean_diff (+0.70) ≈ vjp_delta (+0.70) with identical −C rows. The docstring's framing ("average pullback … does not depend on the persona prompts' own context") is a misconception: the direction is *the persona prompts'* target‑layer contrast. Inference, P≈0.7. **Check:** `cos(v_l, c/‖c‖)` per layer on the saved vector; if >0.7 the method is a mean‑diff relabel.

**2. Result reading: the comparison is inconclusive by construction.**
`index.md`: score = min over sides; −C binds for all three methods (+0.86/+0.90/+0.87, off 0.16–0.20); differences ≤0.04 inside 90% CIs of width ≈1.3; 1 seed → the "hierarchical" bootstrap has zero seed variance. `random` +C gives on‑axis +2.64, i.e. the +C (sycophantic) side is reached by *any* perturbation (degradation reads as agreement), so only −C is discriminative. Dose selection is done on the same 20 questions that are scored (in‑sample optimism, shared by all methods but inflates everything). Column `N` is 20–25 for a 20‑question cohort and `rejected` is unexplained — report what they are. Observed. P≈0.95.

**3. suppressed_mean_diff: the basis is noise and the method as run is mean_diff projected on a ~random 32‑dim subspace.**
`modal-dev-suppressed2.log:113-133`: top tokens are `'atric', 'ůsob', '关灯', 'veda', …`; energy kept 0.008–0.019 vs chance 32/2560=0.0125. Causes in code: `suppressed_mean_diff.py:99-101` scores only the **last token** of prompts that share an identical story suffix (log shows POS/NEG differ only in the persona line 300+ tokens earlier), so the persona contrast of "thought‑but‑not‑said" tokens at that position is ≈0 and `diff.topk` (line 74) picks rare high‑norm unembedding rows. `_contrast_basis` centers `W_U` but never normalizes row norms, so token selection is norm‑biased. Observed + inference, P≈0.9. Correctly excluded from results, but should be reported as "not tested", not "no gain".

**4. Energy‑kept diagnostic uses an isotropic null.**
`suppressed_mean_diff.py:106-107` compares to `rank/d`. The residual stream is anisotropic and `W_U` rows are not uniformly random directions; chance for *this* construction could be well above or below `r/d`. Conclusion (no overlap) is very likely still right. P(null is wrong enough to matter)≈0.2. **Check:** energy kept for 20 draws of 32 random token ids.

**5. Source‑faithfulness points I could not verify (need the source file):**
(a) layer indexing: code maps early/peak 23/25 → blocks 22/24 (`round(frac·n)-1`, line 98), i.e. assumes the source indexes HF `hidden_states` (embeddings at 0). If the source indexes blocks 0‑based, both are off by one. (b) `output = n_blocks-1` pre‑final‑norm, while HF `hidden_states[32]` is *post*‑norm; code re‑applies rmsnorm·gain to all three (line 60), so the output branch gets the gain twice under the HF convention. (c) whether the source centers `rise/fall` across vocab and uses `min(relu,relu)` vs. a persistence rule. Docstring already admits pooling/basis deviations. P(some deviation changes token selection)≈0.5 — moot given #3.

**6. WikiText matching is by token count only, not "same positions".**
`mean_vjp.py:44-52`: contexts are decoded then re‑encoded in `_encode` (token count can drift by ±1–2; harmless), `skip_first=16` now skips wiki tokens rather than the chat header, and contexts are `wikitext-2-raw` with ` @-@ ` / ` , ` artifacts and document‑ordered (first ~400 contexts are a few articles). None of this is a bug, but "generic text" is a narrow, artifact‑laden corpus. P(materially affects result)≈0.15.

## Checked and found nothing
- Delivery parity with `vjp_delta`: same `layers` (6–24 from `resolve_layers`), same `target_layer` default 29 and `--target-layer` plumbing (`walk.py:141`), same cotangent (`_target_mean` last‑token contrast), same `_valid_mask`, same bf16 grad / f32 accumulation, same `_unit_direction`, same `apply` (`y + coeff·v`), same +C = sycophantic orientation (blind judge labels agree).
- `cfg.target_layer` mutation (`mean_vjp.py:73`) survives `Vector(cfg,…)`, `save` metadata and `from_dict`; `vjp_delta` saves `None` instead — cosmetic, apply ignores it.
- Registration in `variants/__init__.py`, `CONFIGS` auto‑discovery in `walk.py`.
- No `cache hit vector` / stale answer reuse in the three logs.
- Pre‑existing (shared, not a differentiator): extraction prompts contain `<think>\n\n</think>\n\n<think>…` (`personas.py:46` prepends `<think>` after the template already emitted an empty think block), while generation uses `enable_thinking=False`.
<!-- saved verbatim by PI/OpenAI; reviewer-anthropic (claude-fable-5-1) fresh-eyes review, 2026-09-26 -->
