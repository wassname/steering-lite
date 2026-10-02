# Evidence review — prompt-gain refinement + dense random reference (2026-10-02), revised after full-log audit

Reviewer: PI/Anthropic (read-only; no shell, no edits, no jobs). Supersedes `evidence-review-initial.md`; the initial report's selection recommendation and "near-null control" wording are withdrawn below.

## 1. Log coverage

`random-walks.log` (15,403 lines) is now read sequentially in full: lines 1–5171 (initial session), 5172–15334 (this session, chunks of 500–800 lines, each under the 50 KB cap), 15330–15403 (initial session; overlaps 15330–15334). **Remaining unread portion: none.** `prompt-walk.log` (438 lines) was read in full initially.

What the newly read range (5172–15334) adds to the stage reconstruction:

| item | observed | lines |
|---|---|---|
| failures / tracebacks / OOM / partial caches | none; every rung `cached=0 missing=20` for fresh seeds | whole range |
| WALK_COMPLETE | s20 03:30:36, s25 03:34:01, s19 03:34:06, s17 03:35:14, s23 03:35:54, s18 03:36:04, s16 03:36:22, s21 03:36:27, s24 03:36:37, s26 03:42:56, s22 03:43:33, s27 03:48:26, s29 03:51:22, s28 03:51:47 (s30/s31 in tail) | 7112–14971 |
| fresh calibrations seen in range | s26 `C0=1.93 start 0.25` (6 iters); s27 `C0=1.424 start 0.1575` (5 iters); s28 `C0=1.772`; s29 `C0=1.885` (13 iters, `n=373` on some probes); s30 `C0=1.715`; s31 `C0=1.815` (9 iters) | 8265, 11562, 11704, 11864, 11956, 12224 |
| stop rule | boundary = two consecutive unhealthy rungs per side, walk ends one rung past both; earliest unhealthy rung C=2 (kl_rms 0.96–1.24, s20/s19), typical 2.52–4 | e.g. 6375, 6418, 5569, 5658 |
| health-rule passes on degenerate text | s22 −C C=5.04/6.35/8 `breakdown=[]` with `mean_words 41.75/5.6/2.5`, output ". total……"; s26 −C C=2.52 "The target is a mid-market SaaS company." (16.3 words); s16 −C C=2.52 numbered two-line answers (95.5 words); s21 +C C=4 `unfinished 10/20` exactly at the 50% threshold → `['unfinished']` | 11205, 11717, 12017, 7245, 7785, 11187 |
| calibration oddities | `rep` column drops 0.96→0.12/0.08 for s31 at c≈1.78–1.86 while the tail text stays coherent; `n=373/396/398/399` prompts on several probes; one trace ran 16 iterations vs banner `<=12 iters` (initial session) | 12003–12012, 11817–11832, 4811–4945 |
| per-t KL profile SHOULD ("decreasing or flat") | spiky, not monotone, on every seed (e.g. t=1 p95 10.13, t=3 11.14, t=5 7.38); advisory only, not asserted | 11793, 11883, 12011 |

Nothing in the range changes the execution verdict: 21/21 random dev walks COMPLETE, 0 FAILED, no partial caches. Timing: fan-out 03:17:02–03:53:41 UTC; fresh seeds ≈ 13–20 min each; s22 ran to 03:43 because its −C side stayed "healthy" by the rule until C=10.08.

## 2. Corrections to the initial report

**(a) Mechanism claim withdrawn.** `scaled_prompt_embeddings` computes `embeddings * torch.where(mask, C, 1.0).to(embeddings)[..., None]`. 2⁻⁸ is exactly representable in bf16 and the product is nonzero, so gain 1/256 is a scaled persona input, not mathematically an absent-persona or zero-embedding control. My "near-null control" and "zero-prefix" framing were inference, not fact. What is observed: output 0 differs between C=0 and C=2⁻¹⁰ and between the two personas at 2⁻¹⁰ (`prompt-walk.log` 03:18:17–03:18:24), and all 14 points at C ≤ 1/32 shift negative on both sides (+C: −0.47…−0.10; −C: −0.84…−0.04). Candidate explanations, none established: H1 low-norm prefix positions change attention/decoding independent of persona; H2 greedy-decoding bifurcation under tiny input perturbation; H3 a small residual persona signal (argued against, not excluded, by both sides moving the same way). Discriminating test: regenerate −C at 1/256 under a different batch composition, and a neutral 10-token prefix at gain 1; same effect under both → H1; unstable → H2.

**(b) No cutoff is proposed.** The initial recommendation to treat gains < 1/16 as controls in `results.py` is withdrawn; selection stays as implemented. The correct distinction is: the table's +0.68 is a valid *conditional in-sample score* under the defined rule (best admissible dose per side, `pareto_score`), and the paired data do not establish a *persona-specific benefit* for the abrasive instruction. From `comparison.json` `selected_controls`: −C at gain 1/256 effect −0.841; the opposite persona at the same gain −0.707; same persona at gain 0 −0.417. The persona-specific difference at the selected gain is −0.134 premise levels; the gain-vs-zero difference is −0.42, shared in sign by the opposite persona. So "prompting_scale −C = +0.84" is a score, not evidence that the abrasive prompt moves the model toward rejection. Both statements belong in the write-up.

**(c) Intermediate support acknowledged.** +C gain 3.75 is an admitted measured intermediate: effect +1.736, mean steered damage 1.367 (≤ 1.5), off_axis 1.08 (`pareto_supports` +C). The initial "no healthy intermediate" was wrong as stated. Remaining gaps on +C: 1/32 (−0.64) → 3/32 (+3.37) with 1/16 rejected; 1.5 (+3.44) → 3.75 (+1.74) with 1.75–3.5 all rejected; then 4.0 (−1.09) and ≥6 bare-like. On −C every gain 3/32–2.75 fails the cap; admitted supports are the eight gains ≤ 1/16 and 3.0–16.

**(d) Identity wording.** Accepted as now corrected by the orchestrator: fresh ordinary vs cached scaled C1, 40/40 identical; same-process fresh check is the 3-prompt `check_prompt_embeddings` (logits and greedy ids exact).

**(e) Cost proxy.** `COST_PROXY 16741.450 s / $9.0739` sums all 22 certificate `total_s`, including the five warmed seeds 11–15 (~50–60 s each); it excludes wrapper, container start/idle, CPU and memory. My earlier "excludes the five cached workers" was wrong. Plan vs observed: 6600 s → 16741 s; fresh random worker ≈ 17 min, not 5.

**(f) Health rule (unchanged P2).** Dotted or two-line answers pass `health()`; the damage cap catches them (s22 −C C=8: `steered_damage 3.867, admissible false`), cost is extra rungs. Observed, not a correctness defect of the scores.

## 3. Status of downstream artifacts (orchestrator-stated vs. verified here)

- Verified here: `full-regression.log` shows `wrote …/full/points.json (2022 points)`, `27b-full (662)`, `olmo-full (348)` each followed by `UAT_PASS`; `final-render.log` shows dev `points.json (1628 points)` and `UAT_PASS`.
- Verified here: `comparison.json` on disk contains `random_11_vs_32` (p90 max bound change 0.32, p75 0.59, p50 0.30 across 11 shared doses), `random_at_two_unfiltered` (64 interventions, 8 negative / 56 positive, mean +1.71, 4 excluded), `selected_controls`, `short_gains`, `pareto_supports`.
- Not verifiable here: `comparison.json` has **no** `per_seed_random_null` key and `comparison.log` ends in `AssertionError: ('full', 'points')` from an earlier `compare.py` revision (the current file asserts `summary`/`blind` and compares points with a blind-field exception). The orchestrator states the rebuild completed (proc_1969 exit 0, 613 s), summaries and selected blind tables are equal, and non-selected points newly attach cached blind ratings with no old rating changed; I have no on-disk output confirming the successful `compare.py` run or the per-seed null numbers. Report both as pending until the refreshed outputs land. Figure-label/tick pipeline: running per orchestrator; not claimed complete.

## 4. Revised findings by priority

- **P1 (interpretation, not selection):** prompting_scale's −C component (+0.84 at gain 1/256) and the +0.68 score are conditional in-sample values; the paired controls (−0.841 vs opposite persona −0.707 vs same-persona gain-0 −0.417) do not support an abrasive-instruction benefit. Write-up should say so beside the table.
- **P1 (gaps):** +C has one admitted intermediate (3.75: +1.74, damage 1.367); 1/16 and 1.75–3.5 are rejected on the cap; −C has no admitted persona-range support between 3/32 and 2.75. The prior "coarse grid hides a gradual transition" is not supported by these data.
- **P1 (null scale, hypothesis):** |effect| up to 0.84 at gains ≤ 1/32 on both sides suggests single-seed/20-question dose-selection noise of that order; the pooled random row (−C +0.11) averages over 32 seeds and is a different statistic. The per-seed random null in `compare.py` is the right check; its output is not on disk yet.
- **P2:** health-rule pass on degenerate short/dotted answers (extra rungs only); calibration solver iterations/`n` variation and `rep` column semantics undocumented; `results.md` Results section still "Pending execution"; `comparison.log`/`comparison.json` on disk are stale relative to `compare.py`.

## 5. Compact ml-debug form

| row | answer |
|---|---|
| log length / config | random-walks 15,403 lines fully read; prompt-walk 438; `GEN` bf16/L40S, transformers 5.12.1, `max_new_tokens 512`, greedy |
| SHOULD lines | fla=True ✓ all workers; chat template ✓; bare healthy ✓; 62/62 prompt rungs healthy; random rungs unhealthy from C≈2; calibration advisory SHOULDs not asserted (per-t profile spiky) |
| null for cited numbers | random pooled −C best +0.11; small-gain prompt points −0.04…−0.84; C=0 −0.47/−0.42; per-seed random null: pending |
| init demo | bare accepts premises (pnf_03) |
| vs dummy | −C selected +0.84 vs opposite persona at same gain +0.71 directed, same-persona gain-0 +0.42 |
| baseline / held-out | single seed, 20 dev questions; no held-out claim made or supported |
| schedule | N/A |
| one full sample | pnf_03 bare/C0 accept; −C 1/256 rejects ("There is no standard \"phase-lock frequency\"…"); mechanism unknown |
| worst step | N/A |
| surprises | both personas negative at tiny gains (chasing: H1/H2/H3 above); s22 health passes on "……" (explained: regex + short-sequence repetition); 16-iter calibration (explained: sampled KL); stale comparison outputs (explained: earlier compare.py revision) |
| missing to trust | per-seed random null output; successful compare.py run log; regeneration of 1/256 under different batch; neutral-prefix control |
| diagnoses | selection-conditional score misread as persona benefit 60%; low-norm-prefix effect 25%; decoding bifurcation 25%; embedding-scaling bug 5% (identity checks pass); unknown 10% |
| cheapest separating test | regenerate −C at 1/256 with shuffled batch order; add neutral 10-token prefix at gain 1 |
| wall-clock | prompt walk 227 s; random fan-out 36.6 min; judging 3.3 min; full rebuild 613 s (orchestrator-stated) |

## 6. Verdict (calibrated)

Execution: credible — every stage completed, no failures, hashes unchanged, exact dev slice, new gains carry the current run_id, judge coverage complete, three full-cohort renders UAT_PASS (highly likely, ~90%). Scientific reading: the table's prompting_scale score is a valid conditional in-sample number under the existing rule and should stay; it should be presented alongside the paired controls, which do not establish an abrasive-instruction benefit (likely, ~65% that the −C best reflects non-persona variation; mechanism unresolved). The refined grid shows one admitted +C intermediate (3.75) and otherwise cap-rejected or bare-like gains; "healthy gradual transition" is not supported. Pending before final claims: per-seed random null output and the successful `compare.py` run record.

Clickable: `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/random-walks.log:12017`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/comparison.json:5753`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/comparison.log:9`, `/workspace/2026/lite/steering-lite-bsbench/src/steering_lite/prompting.py:45`.

```
  ,___,
  (O,O)   "Insufficient skepticism doesn't *feel* like insufficient
  (   )🔍  skepticism from the inside."  -- Nanda, via ml-debug/SKILL.md
   " "
```

— PI/Anthropic