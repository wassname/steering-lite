# Review: user-turn steering, BS-bench full, Qwen3.5-4B (branch dev/prompt-gains-random-reference)

Reviewer: reviewer-anthropic (Claude), read-only. Scope: the named files plus two small deviations I flag below (`src/steering_lite/prompting.py::span_mask`, and a grep of `slop/reviews/2026-10-02_user_turn/learned-walks.log`), because mask correctness and "did every method pass the runtime check" could not be judged without them.

## Verdict

**No blockers in code or data handling.** The position mask does what the report says, decode steps are unsteered, cache methods mask the right dim, sink_split is refused at CLI and runtime, and the walk reuses each method's `_s0` vector and C0 (paired by construction, confirmed in logs for all 18 methods). The plot matches index.md.

**Two presentation/inference items I would fix before anyone quotes the headline** (not bugs, but the claims lean on them):

1. The headline Δ compares against **everywhere seed 0 only**; the published everywhere report is 3 seeds. Whether s0 is a typical everywhere seed for vjp_resid is not shown anywhere in the supplied files. (compare.py:38 `seeds = ... [0]`.)
2. Dose selection is in-sample on both sides, but the user walk has 18 rungs and the everywhere walk fewer; max-over-more-noisy-rungs is more optimistic. A split-half re-analysis costs $0 and is not done.

Spend: results.md reports ≈$48.3 (GPU proxy + Jev), under $50 on its own accounting, with stated exclusions ("Excludes Modal CPU/memory and invoice reconciliation"). I put roughly even odds the real invoice is over $50; this is not a results.md inconsistency.

---

## 1. Plot, fresh eyes first

What I saw before reading index.md (recorded verbatim in my working notes):

- VJP-resid +C ring ≈(2.75, 0.31) is the rightmost low-damage ring; VJP-value +C ≈(2.55, 0.48); sspace_scale +C ≈(2.0, 0.39); mean diff +C ≈(1.45, 0.52); corda_pca +C ≈(1.35, 0.25). Prompt +C star and prompt×gain +C ring at ≈(3.4, 1.2): further right, ~4× the damage.
- Left: sspace_scale −C ≈(−3.4, 0.79), VJP-resid −C ≈(−3.2, 0.52), VJP-value −C ≈(−2.1, 0.6), eng.prompt×gain −C ≈(−1.9, 0.75), mean diff −C ≈(−1.8, 0.95), corda_pca −C ≈(−1.45, 0.22) then a near-vertical dashed drop to its × at ≈(−1.6, 0.76), prompt×gain −C ≈(−1.0, 0.68). eng.prompt×gain +C ring sits at bare.
- Grey random region strongly right-skewed: to ≈+1.7 on the right, only ≈−1.3 on the left, damage 0.2–0.65.
- VJP-resid −C dashed curve visibly broken between ≈−1.4 and ≈−2.6.
- A brown faint dot at ≈(−2.4, 0.35) further left and lower than the eng.prompt×gain −C ring looked like a mis-set ring.

Checks against index.md and code:

| Check | Result | Evidence |
|---|---|---|
| Cohort/counts | ✓ | title "full, 100 questions"; table `seeds`=1 per method, random 16; footer "random: 16 directions" |
| Ring coordinates | ✓ all 7 shown methods | e.g. vjp_resid −C +3.22/0.53, +C +2.77/0.31; prompting_engineered_scale +C +0.04/0.16 at C=0 |
| Curve returns | ✓ none, by construction | results.py:404-409 `smooth_path` only appends points with increasing `directed()`; `frontier(..., include_endpoint=False)` for the path |
| Gap bridging | ✓ not bridged | results.py:416-418 `if h[i] > MAX_PLOT_GAP: ... [None, None]`; the visible VJP-resid −C break is this |
| Random contour = 16 dirs | ✓ | plot footer; results.md "16 directions (s0–s15), not 50" with the budget stop documented |
| Prompt sweeps on same graph | ✓ | plot_marks.json methods include prompting_scale, prompting_engineered_scale |
| The "odd" brown dot | resolved | brown = sspace_scale; its −C jumps −2.4→−3.4 between C=20 and C=32 (gap ≈1.0, not drawn), and the ring is the score-setting dose (on−off = 2.63 > 2.05), not the min-damage point. ~70% confident of the colour attribution from the PNG alone. |

Things a reader of the PNG is not told:
- The plot shows **top-5 of 21** learned methods (results.py:676 `[:TOP_N_PLOT]`). Rule-based, not cherry-picked, but the caption should say "best 5 by score".
- The random zone is **pooled-sign** (results.py:350-358: `chosen = [... for side in ("+C","-C")]`), and it is right-skewed: random-user directions of either sign mostly push toward *accepting* the premise. That makes random a poor symmetric null for −C and means −C wins are competing against a drift the other way. Worth one sentence in results.md.
- Random's score row (−C +1.54/1.09 at C=4, n=300) is pooled over only the **3 of 16 directions still admissible at C=4** (results.py:199 `live = [... if point["admissible"]]`; blind table n=300). It is a survivor-selected statistic plus a dose selection. This inflates random relative to methods (conservative for "methods beat random"), but the table should say "3 directions at this dose".

## 2. Position-mask correctness

Observed in code:

- `positions.py:37-47 select()`: `if length == 1 and mask.shape[1] > 1: return original` → decode steps unsteered; otherwise shape-asserted `torch.where(mask, steered, original)`. Prefill with a different batch shape fails loudly.
- `attach.py:_hook/_linear_hook` wrap every `apply()` output in `select(..., y)` on seq_dim=1 (block outputs `[b s d]`, Linear outputs `[b s *]`).
- Cache methods: `value_gram.py::_edit` → `select(values + coeff*delta, values, seq_dim=2)` on `[b h t d]`; `vjp_value.py::AdditiveValueCache._edit` same with `seq_dim=2`; edits run on incoming `value_states` before cache concat, so decode t=1 → original. `query_steer.py::install` hooks `q_norm` output `[b s h d]`, default seq_dim=1 — correct for Qwen3-style `q_norm(q_proj(h).view(b,s,h,d))`.
- `sink_split.py::_slot_attention`: `assert positions.active() is None, "... no per-token form"`; plus `walk.py:124` CLI assert. The `resid` hook in sink_split_resid does not call `select`, but it is unreachable in user mode because the attention assert fires first (only `attn_off` probes bypass it).
- Mask construction: `prompting.py:18-31 span_mask` — tokens overlapping the first occurrence of `question + suffix`, then `assert tokenizer.decode(ids[selected]).strip() == span.strip()`. `walk.py:214-216 user_spans` = `row["prompt"] + GEN["suffix"]`, no template. Left pad offsets are (0,0) so pads never match (span start > 0).
- Silent-everywhere path: the one I looked for is an `apply()` that edits `y` in place (then `steered is original` and `torch.where` is a no-op). Not detectable statically for 18 methods, but `walk.py:602-628 check_user_positions` asserts `bare[before]==steered[before]` and `not torch.equal(steered, everywhere)` at the start dose of **every** user walk. I count 18 `USER_POSITIONS_CHECK_PASS` lines (17 in learned-walks.log + mean_diff in pilot.log:58), span_tokens=[43,40,47] of prompt_tokens=[55,52,59] — i.e. 12 template tokens excluded. Decode-unsteered is covered by the empty-mask greedy equality (`walk.py:613-614`, `tests/test_pipeline.py:630-632`).
- Paired comparison: `walk.py:122` `args.name = vector_name + "-user"` but `extract_vector` and `calibration_c0` (walk.py:701) use `vector_name` → `vectors/<method>_s0`, `calib/<method>_s0.json`. learned-walks.log shows `cache hit vector .../vjp_resid_s0.safetensors` (line 6174) and for every other method. Same bare answers (`answers/bare/bare.jsonl`, data.py:63). ✓

Gaps (not bugs):
- `rung_kl` skipped in user mode (walk.py:751) — documented; means no mechanical dose diagnostic for user walks, only Jev.
- Start at C0/2 (`--start-below 2`, pilot.log:39). For vjp_resid irrelevant (best C=1 and 0.794 vs start 0.125). For methods that got *worse* under user-turn, I cannot tell from the supplied files whether their best C equals the start rung (linear_act 1.26, chars 1.26, spherical 0.0496, pca 0.397). If so the floor may have clipped the optimum. Check: compare `best C` to `GRID[start]` in each `walks/<m>-user_s0_full.json`.

## 3. Headline calibration

Observed: comparison.md vjp_resid user +2.47 vs everywhere s0 +0.75, Δ +1.72 [+0.94, +2.02]; sspace_scale Δ +1.73 [+0.96, +2.22]; six methods worse with CIs excluding 0 for linear_act, chars, topk, value_gram, pca. Audit −C: vjp_resid .87/.20, mean_diff .54/.26, random .50/.30, bare .89/.52. All numbers agree between index.md, comparison.md and results.md.

The caveats listed in results.md (one seed, in-sample dose, 16 random dirs, C0/2 start, 18-rung cap, direction-only replication of the other agent) are the right ones and are stated. What I would add or sharpen:

- **Seed-0 everywhere baseline.** Δ is paired on questions but the everywhere side is one seed of a 3-seed report. If vjp_resid s0 everywhere is below its seed mean, Δ is inflated. The parent can check `outputs/bsbench/results/full/index.md` (not in my scope). Until then the honest statement is "+1.72 vs that one vector steered everywhere", not "vs the everywhere method".
- **What the −C "win" is, mechanistically.** examples-vjp_resid-neg.md: steered answers open with "There is no such concept…", "I cannot answer this question because the scenario you described does not exist…". These are refusals-with-reason; Jev premise → ~0. The audit separates *generic/off-target* refusals (mean_diff "the SaaS target does not exist", fin_fa_02 "you cannot use the provided text" → on_target .01) from on-topic ones. It does **not** separate "rejects for the real flaw" from "rejects for a hallucinated flaw": sw_af_02 steered answer "**Net** is not an IDE theme" (misread "Net of") gets on_target .85, fabricates .05, premise 6.99→0.09. So on-axis for −C still conflates (a) finding the real flaw, (b) contrarian refusal, (c) invented flaw; audit removes most of (b) only. Inference, moderate confidence (~65%): a non-trivial share of the vjp_resid −C gain is (b)/(c), because steering only the question tokens is a natural way to make the model treat *the question* as suspect regardless of content.
- **Alternative explanations, ranked by how much they could move the headline:**
  1. Contrarianism rather than discernment (above). Not tested: the bench has no sound-premise controls.
  2. Single seed. Other agent's result points the same direction (quoted, unverified) — weak corroboration.
  3. In-sample dose selection with 18 vs fewer rungs. Likely a few tenths at most, but asymmetric.
  4. Batch-composition drift (35–56 % text mismatch, prompt-walks.log:123-244): symmetric in premise, but it puts a positive floor on |Δdamage| (~0.15–0.2 at low dose for every method, visible in the plot). Depresses all absolute scores, not Δ.
  5. Judge-correlated errors: aware ratings, blind, and audit are all Jev; audit bare on_target .89 implies ~10 % false "off-target" floor; the fabricated-rejection probe got on_target .61 when it should be ~1 (it *does* address the request), so on_target and fabricates are entangled.
- **Cheapest discriminating next steps** (in order):
  1. $0: split-half dose selection on points.json (select dose on 50 questions, score on the other 50, both views) — removes in-sample optimism from Δ.
  2. ≈$1–2 + Jev: run vjp_resid-user at C=1 (−C) and everywhere at its best −C dose on ~20–50 **sound-premise** questions (e.g. the `ood` cohort if it is sound-premise, or hand-written); measure false-rejection rate vs bare. If user-turn rejects sound premises far more, the win is contrarianism.
  3. ≈$3–5: vjp_resid-user seeds 1–2 only, to put seed variance on the headline.

## 4. Audit request design

`judge.py:122-136`: two yes/no choices, each naming its false case; separate request so existing ratings' cache keys are unchanged (good). The `on_target` "no" criterion explicitly targets the safety-refusal failure mode, which matches the observed failure in mean_diff/random examples.

`audit_probe.py`: three hand-written answers on **one** question (`list(c)[0]`), written by the same agent that wrote the rubric. Results per results.md: clean .99/0, fabricated .61/1, off-target 0/0. This is a smoke test that the two probabilities move in the right direction; it is not validation. Missing cases that matter here: (i) a bare-style on-topic *accepting* answer (should be on_target ≈1 — the bare mean of .89 suggests it is not quite), (ii) a rejection for an invented flaw (sw_af_02 above), (iii) a borderline "your scenario doesn't exist" refusal when the term is indeed fake (med_st_01 got .99 — defensible, but untested by design). The relative comparison vjp .87 ≈ bare .89 ≫ mean_diff .54 is still informative because the floor is shared.

## Nits

- results.py:676 — caption should say the plot shows the 5 best-scoring methods of 21.
- index.md random row — add "(3 directions admissible at this dose)" or show per-dose seed counts; results.py:194-209.
- index.md intro — note the random zone is pooled-sign and right-skewed.
- prompting_engineered_scale +C "best" is C=0 (instruction zeroed): a degenerate optimum; worth a footnote rather than a −0.12 row read as "engineered prompt hurts".
- results.md "Checks before the run" lists 6 tests passed, but the test only covers 6 methods; the runtime check is what covers all 18 — say so (it is the stronger evidence).
- tests/test_pipeline.py:612-636 does not test a *non-empty* mask under cached decode (only the empty-mask case); `select()` handles it identically, but one assertion `generate(mask) != generate(empty)` and `steered_decode_logits == f(prefix)` would close the gap.

## Deviations from "read only the named paths"

- Read `src/steering_lite/prompting.py:18-31` (`span_mask`), called from walk.py:237/608: required to judge whether the mask covers only user content.
- Grepped `slop/reviews/2026-10-02_user_turn/learned-walks.log` for `USER_POSITIONS_CHECK_PASS|WALK_COMPLETE|cache hit vector` to confirm the 18-method runtime check and the `_s0` vector reuse for vjp_resid. No other files touched.

Files the parent should open: `/workspace/2026/lite/steering-lite-bsbench/outputs/bsbench/results/user-full/plot.png`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_user_turn/examples-vjp_resid-neg.md` (sw_af_02 and med_st_01 entries), `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/results.py:194`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_user_turn/compare.py:38`.

-- reviewer-anthropic (Claude)