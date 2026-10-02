# Evidence review — prompt-gain refinement + dense random reference (2026-10-02), final

Reviewer: PI/Anthropic (read-only; no shell, edits, jobs, or paid calls). Third and final version; the initial and revised reports are preserved by the parent. This version keeps the full-log coverage and the withdrawn mechanism/cutoff corrections, and adds verification of the completed comparison and render passes.

## 1. Log coverage (unchanged from revised report)

`random-walks.log` 15,403 lines read sequentially in full across sessions (1–5171; 5172–15334; 15330–15403). `prompt-walk.log` 438 lines in full. **No unread portion.** Execution: 21/21 random dev walks COMPLETE, 0 FAILED/Traceback, no partial caches; prompt walk 31 gains × 2 sides healthy by the mechanical rule; identity checks pass.

## 2. Standing corrections (kept)

- Gain 1/256 is a nonzero bf16 scale (`torch.where(mask, C, 1.0).to(embeddings)`), not an absent-persona control. The initial "near-null/zero-prefix" framing is withdrawn. Both personas shift negative at gains ≤ 1/32 (observed); low-norm-prefix, decoding-sensitivity and residual-instruction explanations remain competing hypotheses, none established, no neutral-prefix or fixed-dose repeat run.
- No admissibility cutoff below 1/16 is proposed; production selection is unchanged.
- +C gain 3.75 (+1.736, mean damage 1.367, off_axis 1.080) is an admitted measured intermediate; gaps remain (1/16 rejected; 1.75–3.5 rejected; −C has no admitted support between 3/32 and 2.75).
- Identity wording: fresh ordinary vs cached scaled gain-one 40/40; same-process fresh check on 3 prompts.
- Cost proxy sums all 22 certificate totals including warmed seeds 11–15; excludes wrapper/startup/CPU/memory.

## 3. Newly verified on disk

**`compare.py` + `comparison.log` (current run).** Quoted:

> `ADDITIONAL_CACHED_BLIND_ATTACHMENTS full 319`
> `FULL_SCIENTIFIC_REGRESSION_PASS full keyed points, scores/selection, selected blind ratings, raw random bounds unchanged; CI changes 0 ; random seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
> `HISTORICAL_BOOTSTRAP_REPLAY_PASS 27b-full old CI endpoints reproduced exactly using pre-rename ordering [...]`
> `HISTORICAL_BOOTSTRAP_REPLAY_PASS olmo-full old CI endpoints reproduced exactly using pre-rename ordering [...]`

What the code actually asserts (read, not taken from the message): `before["blind"] == after["blind"]`; every summary row equal except `ci`/`ci_room`; keyed points `(method, seed, side, C)` equal in all non-`questions` fields; per-question fields equal except `blind`, where only `None → rating` is allowed; raw zone bounds equal; random seeds equal. For reports with CI differences it rebuilds `choose(after["points"])`, iterates methods in the pre-rename order (`vjp_resid→vjp_delta`, `vjp_value→vjp_cache`) with a fresh `random.Random(0)`, calls production `bootstrap`, and asserts exact equality with the saved endpoints. That is a genuine replay, not an alias in production. Observed CI shifts in the new order: 27B max |Δ| 0.0226 (vjp_resid upper 0.2525→0.2752), OLMo max 0.0183 (vjp_resid upper 0.0339→0.0522); 4B full: 0 changes. The "≤ ~0.023" statement is supported.

**`comparison.json` → `per_seed_random_null` (32 entries).** Recomputed from the printed list: per-direction `score` range −0.495 (s20) … 0.9875 (s7); `negative_side_score` sorted median = (0.3095 + 0.327)/2 = 0.31825; six seeds ≥ 0.6795 on both `score` and `negative_side_score` (s31 0.687, s18 0.6965, s12 0.778, s30 0.899, s10 0.968, s7 0.9875). Caveat stands and should stay in the write-up: this is descriptive — random directions are real iso-KL interventions with a different dose grid and walk length than the 31-gain prompt sweep; it is not a p-value or a matched search.

**Dose-2 random asymmetry.** `new_zones` p90 row at C=2: `seed_counts 28`, `negative_counts 7`, `positive_counts 49`, `mean_effect 1.7370`, center 2.06925; unfiltered 64: 56/8, mean 1.7107. The results.md numbers match.

**`final-render.log`:** five `wrote …/points.json` lines (dev 1628, prompt-dev 1628, full 2022, 27b-full 662, olmo-full 348) each followed by `UAT_PASS`; PNG/page frontier-mark counts agree (77/77, 29/29, 91/91, 72/72, 63/63).

## 4. User-facing wording check (`results.md`, current)

| requirement | status | evidence / needed edit |
|---|---|---|
| persona-specific evidence vs conditional score | ✓ | "Conditional in-sample scores…"; "Most of this selected shift is shared across personas. This does not establish a benefit from the abrasive instruction." |
| mechanical health vs damage cap | ✓ | "All prompt mechanical-health checks pass; the existing mean-damage cutoff excludes 27/62 short and 7/18 engineered points." Random s22 near-empty outputs noted in the form. |
| actual vs planned cost | ✓ | 16741.450 s, $9.0739 GPU-only, $9.5088 with Jev, exclusions listed, "original 6600-second/$3.58 forecast underestimated". |
| Monte Carlo (bootstrap) differences | ✗ stale | Text still says "The final replay result remains pending." `comparison.log` now shows `HISTORICAL_BOOTSTRAP_REPLAY_PASS` for 27b-full and olmo-full and `CI changes 0` for 4B; update to state the replay passed, max shift 0.0226 (27B) / 0.0183 (OLMo), 4 methods each, 4B unchanged. |
| no held-out claim | ✓ | "This is not held-out or multiseed prompt evidence." |
| per-seed random descriptive comparison | ✗ absent | results.md's "Null scale" row mentions the diagnostic exists but gives no numbers; add range −0.495…0.9875, median 0.31825, 6/32 ≥ 0.6795, with the "descriptive, not a p-value or matched search" caveat. |
| render / review status | ✗ stale | "Final rendering is re-running after review", "Initial dev/prompt builds and three full-report builds passed browser UAT", "Final comparison/figure re-check remains pending" → five final builds `UAT_PASS`; comparison complete. Checklist item 5 (visual re-check, local commit) still legitimately open. |
| minor | – | `random-walks.log: 15402 newline-terminated lines` vs 15,403 lines as read (last line likely unterminated); harmless. "This resumed run does not fresh-generate scaled gain-one for all 40 cases" is accurate. |

## 5. Remaining P2 observations (unchanged)

- `health()` admits near-empty/dotted answers (s22 −C C=5.04–8; s26/s16 −C C=2.52); damage cap rejects them; cost is extra rungs.
- Calibration: 13–16 bisection iterations vs banner `<=12`; `n=373–399` probe counts and the `rep` column semantics undocumented.
- Neutral-prefix control and a fixed-dose repeat are the cheapest discriminators for the small-gain shifts; not run.

## 6. Final verdict

Execution and provenance: credible (almost certain, ~90%): full logs read, no failures, 568 hashes unchanged, exact dev slice, new gains carry the current run_id, judge coverage complete, all five reports re-rendered with UAT_PASS, keyed points/scores/selections/existing blind ratings/raw random bounds equal across the rebuild, 319 previously-absent blind ratings attached without altering any existing rating, and the larger-model CI endpoint shifts explained by an exact replay under the historical RNG allocation order.

Scientific reading: the short-prompt score 0.6795 is a valid conditional in-sample value under the existing rule and may stand in the table; it is not evidence of an abrasive-instruction effect (paired controls −0.841 vs −0.707 opposite persona vs −0.417 gain-0), and six of 32 single random directions reach or exceed it under the same per-direction selection — a descriptive comparison only. The refined grid shows one admitted +C intermediate (3.75) and otherwise cap-rejected or bare-like gains. No held-out or multiseed prompt claim is made or supported. Three wording updates to `results.md` (bootstrap replay status, per-seed numbers with caveat, final render status) are needed before it is current; nothing else blocks the local commit.

Clickable: `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/comparison.log:1`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/compare.py:66`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/results.md:68`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/final-render.log:498`.

```
  ,___,
  (O,O)   "If you didn't fix it, it ain't fixed."
  (   )🧪      -- Agans' ninth rule, via ml-debug/SKILL.md
   " "
```

— PI/Anthropic