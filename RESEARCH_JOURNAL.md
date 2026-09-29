# Research journal (steering-lite-bsbench)

Oldest first. Entries by agents are signed.

## 2026-09-27 -- Steering on larger models: VJP methods fail on both 64-layer models, calibration checked

This entry records what happened when the BS-bench v2 steering protocol was moved from Qwen3.5-4B to two larger models, and which explanations the checks ruled out.

Setup, same for all three models: 100 BS-bench v2 questions, greedy decoding, 2-sentence answers, bf16, persona contrast pairs, steered layers = middle 60% of depth, Jev judge (premise level 0 to 8, damage 0 to 4). "on-axis / room" is the on-axis premise change at the Pareto-best dose divided by how far the bare answers could still move toward that side (8 minus bare level for +C, bare level for -C), taking the weaker side. Commit `a75d69f`.

| method | Qwen3.5-4B score | 4B on-axis / room | Qwen3.5-27B score | 27B on-axis / room | OLMo-2-32B score | OLMo on-axis / room |
|---|---|---|---|---|---|---|
| mean_diff | +0.37 [+0.15, +0.78] | +0.14 [+0.08, +0.24] | +0.91 [+0.60, +1.33] | +0.64 [+0.50, +0.76] | +0.21 [+0.04, +0.50] | +0.27 [+0.21, +0.33] |
| chars | +0.88 [+0.49, +1.23] | +0.35 [+0.22, +0.43] | +0.43 [+0.01, +1.11] | +0.36 [+0.07, +0.45] | +0.13 [-0.04, +0.34] | +0.23 [+0.11, +0.27] |
| vjp_cache | +1.14 [+0.75, +1.56] | +0.40 [+0.30, +0.48] | +0.34 [+0.09, +0.69] | +0.33 [+0.19, +0.47] | -0.20 [-0.38, -0.10] | +0.00 [-0.04, +0.04] |
| vjp_delta | +0.66 [+0.39, +1.14] | +0.24 [+0.18, +0.39] | -0.01 [-0.13, +0.25] | +0.13 [-0.01, +0.25] | -0.05 [-0.16, +0.03] | +0.03 [-0.01, +0.05] |
| vjp_delta, target layer 48 | | | +0.21 [-0.02, +0.46] | +0.23 [+0.12, +0.35] | | |
| random | -0.07 [-0.22, +0.13] | +0.01 [-0.03, +0.06] | -0.05 [-0.12, +0.14] | +0.02 [-0.03, +0.13] | -0.10 [-0.15, -0.02] | +0.01 [-0.01, +0.07] |

Table 1. Score = min over sides of (on-axis minus off-axis) at each side's Pareto-best dose, with 90% bootstrap CI. Seeds: 4B and 27B 3 per learned method, OLMo 1, target-48 run 1. Source: `slop/reviews/2026-09-26_27b/compare.md:11-16`.

What the checks showed, each with its source:

- The dose walks reach incoherence and sample around it. Over 130 (4B), 40 (27B) and 14 (OLMo) walk sides, every side has admissible doses before breakdown and broken doses after it, and no dose is flagged broken while Jev damage is below 1.0 (`slop/reviews/2026-09-27_27b_debug/calib_audit.md:2,137,182`). Breakdown is real degeneration: vjp_delta -C on 27B goes from 60 to 280 words per answer and from 0% to 67% unfinished answers, with a bare unfinished rate of 0 to 1% (measured this session from the walk certificates, script `slop/reviews/2026-09-27_27b_debug/calib_audit.py`). The best dose on the -C side, which sets every Qwen score, is never the last admissible dose before a sampling gap.
- The -C side of vjp_delta on 27B never moves toward candour: at its best dose C=0.5 the -C on-axis is +0.22, and the next dose breaks (`slop/reviews/2026-09-27_27b_debug/walk_audit.md:107,109`). On 4B the same method reaches +1.16 before breaking (same file, 4B vjp_delta block). mean_diff on 27B reaches +1.26 on the same questions.
- The bare 27B answers are much more candid: mean premise level 1.92 against 4.04 on 4B, 69% against 37% rejecting (`slop/reviews/2026-09-27_27b_oracle/brief.md:23-24`), confirmed on raw answers (4B: "Calculate the beat frequency by determining the least common multiple of your frontend and backend deployment cycles", Jev level 6.99; 27B: "Beat frequency and standing wave patterns are physical concepts from wave mechanics that cannot be mathematically applied", Jev level 0.02).
- Extraction is not the problem. Batch size 2 against 8 gives per-layer cos of at least 0.997 (`outputs/logs/batch-invariance.log:59,115,169`). Seeds give nearly the same vector (cross-seed cos 0.999 for vjp_delta on 27B), and the 27B vjp_delta vector is not concentrated in a few outlier channels (top-1 coordinate share 0.007, against 0.017 for mean_diff) (`slop/reviews/2026-09-27_27b_debug/vector_geometry.md:7-8`).
- Moving the vjp_delta target layer on 27B from 61 (3 from the end, the default) to 48 raises the -C peak from +0.22 to +0.44 (`slop/reviews/2026-09-27_27b_debug/walk_audit_t48.md:77`) and on-axis / room from +0.13 to +0.23 (Table 1).

Interpretation (mine, PI/Claude): the failure is specific to the VJP methods and appears on both 64-layer models, including OLMo-2-32B, which is not distilled, while mean_diff and chars carry over. That makes distillation-driven superposition an unlikely main cause (maybe 0.1). The target-48 run points at the VJP target rule, "3 layers from the end", as a probable cause (maybe 0.5): on 4B that is 91% of depth, on the 64-layer models 95%, which may sit inside the final stage where features are suppressed (Lad, Gurnee and Tegmark 2024 call it residual sharpening, arXiv 2406.19384). This rests on one seed with overlapping CIs. The alternative that the final-stage band is an absolute number of layers (Wendler et al. 2024 describe Chinese spiking "on the last five layers" in Llama-2 7B, 13B and 70B) is not excluded; a relative rule and an absolute rule predict different best targets on 64 layers (roughly 48 against 58).

Next: small single runs only, wassname's instruction. First candidate: measure, on each model, where the persona contrast peaks and then shrinks toward the output, to place the VJP target before that point, then one walk per model at that target.

The main lesson is that the large-model failure is a property of how the VJP methods pick their target, not of the dose sweep or the extraction code.

-- PI/Claude

## 2026-09-28 -- Where the persona contrast forms: not a fixed fraction of depth

This entry answers wassname's question whether the right VJP target layer is a fixed percentage of depth, using a forward-only measurement on the three models.

`walk.py --profile` records, per layer and at the last token of the 200 persona contrast pairs, ratio = |mean(h_pos) - mean(h_neg)| / mean |h|, i.e. how large the persona contrast is relative to the residual at that depth. No steering is applied.

| model | layers | peak layer (depth) | depth at 50% of peak | depth at 90% of peak | layers from end at 90% | steered band (depth) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 32 | L30 (0.97) | 0.52 (L16) | 0.61 (L19) | 12 | L6-L24 (0.19-0.77) |
| Qwen3.5-27B | 64 | L62 (0.98) | 0.78 (L49) | 0.98 (L62) | 1 | L12-L50 (0.19-0.79) |
| OLMo-2-32B | 64 | L47 (0.75) | 0.65 (L41) | 0.71 (L45) | 18 | L12-L50 (0.19-0.79) |

Table 1. Source: `slop/reviews/2026-09-28_depth_profile/profile_compare.md:7-9`, computed from `outputs/logs/profile-{4b,27b,olmo}.log`; plot `slop/reviews/2026-09-28_depth_profile/profile_compare.png`.

The contrast forms at depth 0.52 to 0.61 on 4B, 0.65 to 0.71 on OLMo and 0.78 to 0.98 on Qwen 27B, so neither a fixed fraction of depth nor a fixed number of layers from the end lines up across the three models. On OLMo the contrast falls after its peak (to about 0.87 of peak at the default VJP target, L61); on the two Qwen models it stays near its peak to the end.

Interpretation (mine, PI/Claude): layer settings probably need to come from a per-model measurement rather than from one rule (probable, maybe 0.7). On Qwen 27B the concept forms at the edge of the steered band, so most source layers lie before it exists, which is a plausible reason the VJP linearisation works poorly there. On OLMo the default target sits where the contrast is already falling, which fits wassname's suppression idea, but the default target is at 0.87 of peak or more on all three models, so the target position alone does not separate them. The measure is a norm ratio, not the rise-and-fall logit-lens rule from wassname's suppressed-activations repo, and it uses one seed of pairs.

Next: one single run, OLMo vjp_delta with the target at its contrast peak L47 (sources L12 to L46). If a target in the falling zone is what breaks VJP on OLMo, this run should recover a clear effect there.

The practical lesson is to measure where a concept forms in each model before choosing where to steer and where to aim.

-- PI/Claude

## 2026-09-28 -- OLMo vjp_delta with its target at the contrast peak: no recovery

This entry records the single run that tested whether aiming the VJP target at the layer where the persona contrast peaks, instead of three layers from the end, brings vjp_delta back on OLMo-2-32B.

| OLMo-2-32B vjp_delta, 1 seed | score [90% CI] | on-axis / room [90% CI] |
|---|---|---|
| target L61 (default), sources L12 to L50 | -0.05 [-0.16, +0.03] | +0.03 [-0.01, +0.05] |
| target L47 (contrast peak), sources L12 to L46 | -0.08 [-0.12, -0.05] | +0.00 [-0.01, +0.03] |

Table 1. Source: `slop/reviews/2026-09-26_27b/compare.md:14,16` (OLMo columns).

Per dose, the target-47 walk moves the premise by at most +0.14 on -C before damage rises (`slop/reviews/2026-09-27_27b_debug/walk_audit_t47.md`, OLMo vjp_delta-t47 block), the same pattern as the default walk in the same file.

Interpretation (mine, PI/Claude): the prediction failed, so on OLMo the target position is very probably not what breaks vjp_delta (maybe 0.85), and the depth profile from the previous entry does not explain the VJP failure. My next suspect is the linearisation itself: the VJP direction is a first-order prediction of what changes the target-layer contrast, and across 30 to 50 layers of a large model that prediction may not hold at the doses used. A direct check is to steer along the VJP direction and measure whether the target-layer activation actually moves along the target contrast, on 4B (where VJP works) against OLMo and Qwen 27B, with mean_diff as a reference.

The takeaway is that moving the VJP target does not fix VJP on OLMo, so the cause lies elsewhere.

-- PI/Claude

## 2026-09-28 -- VJP mechanism check: same signature on all three models, including 4B

This entry records a forward-only check of whether the VJP vectors move the target-layer activation along the persona contrast that defines them, run to separate a size-specific bug from a real limit.

For each model and method (seed-0 vectors), steering at C = +-C0/8 and the target-layer residual at the last token are compared with c = mean(h_pos) - mean(h_neg) of 32 held-out persona pairs at that layer (the VJP cotangent). cos is the cosine between the mean shift and c.

| model | method | cos on benchmark prompts, +C / -C | cos on negative-persona prompts, +C / -C |
|---|---|---|---|
| Qwen3.5-4B | mean_diff | +0.443 / -0.449 | +0.759 / -0.751 |
| Qwen3.5-4B | vjp_delta | -0.067 / +0.077 | -0.499 / +0.474 |
| Qwen3.5-4B | vjp_cache | +0.011 / +0.016 | -0.445 / +0.385 |
| Qwen3.5-27B | mean_diff | +0.291 / -0.278 | +0.569 / -0.525 |
| Qwen3.5-27B | vjp_delta | +0.001 / -0.003 | -0.347 / +0.358 |
| Qwen3.5-27B | random | +0.023 / +0.005 | -0.012 / +0.046 |
| OLMo-2-32B | mean_diff | +0.461 / -0.455 | +0.610 / -0.582 |
| OLMo-2-32B | vjp_delta | +0.090 / -0.077 | -0.458 / +0.458 |
| OLMo-2-32B | random | +0.047 / -0.041 | +0.065 / -0.066 |

Table 1. Source: `slop/reviews/2026-09-28_vjp_check/vjp_check.md` (full table incl. vjp_cache on 27B and OLMo and the t48 and t47 variants), raw rows in `outputs/logs/vjpcheck-{4b,27b,olmo}.log`. 4B random was killed on Modal (exit -9) and not rerun.

Two things hold on every model, 4B included. On benchmark prompts the VJP vectors barely move the target layer along c (|cos| at most 0.09, about the size of random), while mean_diff does (cos 0.29 to 0.46). On the negative-persona prompts, +C moves the target layer against c (cos -0.35 to -0.50) for both VJP methods, and the t48 and t47 variants show the same sign.

Interpretation (mine, PI/Claude): the check does not separate 4B, where VJP works on the judge, from the two large models, where it does not. So a bug that only shows on deep models is unlikely (maybe 0.15), and so is "the first-order VJP prediction breaks with depth" (the prediction already fails on 4B). The anti-alignment is not a contradiction of the code: vjp_delta is mean_pos(J^T c) - mean_neg(J^T c), a difference of gradients between the two classes, not the gradient of c . h, so nothing requires it to raise c . h. My read is that the VJP vectors do not steer through the target-layer persona contrast at all, which also fits the small effect of moving the target (t48, t47). Why they move the answer on 4B and not on the large models stays open.

The takeaway is that the target-layer view of VJP does not explain its success on 4B, so tuning the target is unlikely to fix the large models.

-- PI/Claude

## 2026-09-28 -- Judged candour effect by bare stance: Qwen 27B is mostly the floor, OLMo is a real VJP failure

This entry splits the judged -C effect by whether the bare answer already rejects the premise, after wassname pointed out that the judged steering effect, not a target-layer proxy, is what should decide the question.

At each method's -C Pareto-best dose, the judged candour gain per question is the bare Jev premise level minus the steered level (0 to 8 scale), split by the bare answer: accepts (level 6 or more), middle, rejects (level 1 or less). n counts question-seed pairs.

| model | method | bare accepts: n, gain | middle: n, gain | bare rejects: n, gain |
|---|---|---|---|---|
| Qwen3.5-4B | mean_diff | 153, +1.24 | 42, +0.80 | 105, -0.53 |
| Qwen3.5-4B | vjp_cache | 153, +2.64 | 42, +2.05 | 105, -0.11 |
| Qwen3.5-4B | vjp_delta | 153, +1.87 | 42, +0.98 | 105, -0.31 |
| Qwen3.5-27B | mean_diff | 63, +4.28 | 30, +3.07 | 207, +0.04 |
| Qwen3.5-27B | vjp_cache | 63, +2.86 | 30, +1.38 | 207, -0.14 |
| Qwen3.5-27B | vjp_delta | 63, +1.41 | 30, +0.37 | 207, -0.11 |
| Qwen3.5-27B | random | 21, +0.24 | 10, -0.06 | 69, -0.02 |
| OLMo-2-32B | mean_diff | 85, +1.91 | 11, +0.82 | 4, -0.31 |
| OLMo-2-32B | chars | 85, +1.63 | 11, +1.03 | 4, -1.06 |
| OLMo-2-32B | vjp_cache | 85, +0.11 | 11, -0.52 | 4, -0.42 |
| OLMo-2-32B | vjp_delta | 85, +0.15 | 11, +0.19 | 4, +0.17 |
| OLMo-2-32B | random | 170, +0.03 | 22, +0.33 | 8, +0.04 |

Table 1. Source: `slop/reviews/2026-09-28_judged_by_stance/by_stance.md` (script `by_stance.py` in the same folder, reads `outputs/bsbench/results/{full,27b-full,olmo-full}/points.json`); chars on Qwen and the t48 and t47 variants are in the same file.

On Qwen 27B, on the questions where the bare model still accepts the premise, vjp_cache moves the premise by +2.86 levels (4B +2.64) and vjp_delta by +1.41 (4B +1.87). Only 21 of 100 bare 27B answers accept the premise, against 51 on 4B, so the averaged score is dominated by questions with no room. On OLMo, 85 of 100 bare answers accept the premise, mean_diff and chars move them by +1.91 and +1.63, and both VJP methods stay near random (+0.11, +0.15; random +0.03).

The +C side agrees on Qwen 27B, where most bare answers leave room toward accepting: vjp_delta +C on-axis +4.78 (off-axis 0.61), vjp_cache +4.94 (0.57); blind stance shift +1.37 and +1.47, P(accepts_premise) 60% and 68% (outputs/bsbench/results/27b-full/index.md, method table and blind table).

Caveats (review round 1, slop/reviews/2026-09-28_review/). n counts question-seed pairs, not independent questions: the 27B accepts bucket is 21 unique questions over 3 correlated seeds (4B 51 x 3, OLMo 85 x 1). The table has no CI. The dose was chosen on the same 100 questions it is evaluated on. The bare-accept subsets differ across models (21 vs 51 vs 85 questions), so the split checks headroom within each model; it is not a controlled cross-model comparison. Selecting questions on one bare rating can also regress to the mean.

Interpretation (mine, PI/Claude): on Qwen 27B the VJP methods probably still work where there is something to steer (probable, maybe 0.75), and most of their score drop is the candour floor; mean_diff's lead there is real though (+4.28 on the same questions). OLMo is the case where VJP genuinely fails with plenty of room. OLMo differs from Qwen in family and architecture (plain attention, norm after each sublayer), so a VJP-specific interaction with the OLMo architecture, or a bug that only shows there, is now the leading question; the earlier "fails on 64-layer models" framing was too broad.

The takeaway is that the Qwen 27B drop is mostly a property of the benchmark on a more candid model, and the open failure is VJP on OLMo.

-- PI/Claude

## 2026-09-28 -- VJP on OLMo: not precision noise, not the <think> prompt mismatch

This entry records two checks of why vjp_delta steers Qwen but not OLMo-2-32B, where 85 of 100 bare answers accept the premise.

Split-half stability (`walk.py --vjp-split`, outputs/bsbench/<model>/vjp_split/vjp_delta_s0.json): vjp_delta extracted from pairs 0-99 and from pairs 100-199 agree, median cos over source layers 4B +0.993, Qwen 27B +0.986, OLMo +0.985. The class difference is not a small remainder: |pos - neg| / |pos| median 1.67 / 1.37 / 1.29. So bf16 cancellation noise is ruled out on all three models.

Judged -C walk, seed 0 (outputs/bsbench/results/olmo-full/points.json): OLMo vjp_delta on-axis stays between -0.01 and -0.22 up to KL 1.9, then breaks; mean_diff reaches -1.70 while admissible. For comparison, 4B vjp_delta reaches -1.16 at KL 0.65 (outputs/bsbench/results/full/points.json). Read by hand, OLMo vjp_delta answers change format (a leading ">" quote, restating the question), not stance. The OLMo persona contrast peaks at L47 (slop/reviews/2026-09-28_depth_profile/profile_compare.md), the target of the failed t47 run, so the target layer is ruled out as well.

`<think>` mismatch (flagged independently by the j-steer session): extraction pairs put a literal "<think>" before the suffix, eval runs thinking off, and OLMo has no thinking mode. vjp_delta averages gradients over all prompt positions, mean_diff reads only the last token. Walk with `--no-think --tag nothink` (outputs/logs/modal-olmo-vjp_delta-nothink.log, judge-olmo-nothink.log, results-olmo-nothink.log): score -0.12 [-0.22, -0.00], on-axis / room +0.00 [-0.02, +0.05], same as default vjp_delta (-0.05, +0.03). Vector cos to the default vector per layer: min 0.979, median 0.995 over 39 source layers (slop/reviews/2026-09-28_judged_by_stance/vector_cos_nothink.md, script vector_cos_nothink.py in the same folder). The "<think>" text barely changes the vector.

Interpretation (PI/Claude): no bug found in 5 checks (calibration, batch invariance, precision, target layer, prompt mismatch). The likely remaining explanation (probable, ~0.65) is that the VJP direction on OLMo is a stable, high-gain direction for the late-layer persona contrast that acts on format rather than premise stance; this is a method limit on OLMo, not a code fault. Unknown unknowns keep the rest.

-- PI/Claude

## 2026-09-28 -- Correction: OLMo VJP answers are rewordings, not "format" changes

The previous entry said OLMo vjp_delta answers "change format, not stance". That rested on 3 answers read by hand. A count over all 100 answers at the -C strongest admissible dose (seed 0) gives the following (share of answers starting with ">"; mean words, bare 34):

| OLMo -C, strongest admissible | ">"-start | words |
|---|---|---|
| mean_diff C=2 | 0% | 14 |
| vjp_delta C=0.397 | 78% | 63 |
| vjp_delta-nothink C=0.397 | 1% | 43 |
| vjp_cache C=3.17 | 0% | 137 |
| random C=5.04 | 0% | 42 |

The ">" prefix appears only in default vjp_delta, and only at its strongest dose. On 4 random questions read side by side with bare, vjp_delta-nothink rewords the bare answer and keeps accepting the made-up premise ("ISO 32170", "ABA Model Standard 4.7"); vjp_cache makes the answer longer and adds hedges ("without more context"), and still accepts it. mean_diff shortens answers and rejects ("Neither is standard"). The better description: on OLMo, VJP changes wording and length, not premise stance, up to breakdown.

-- PI/Claude

## 2026-09-29 -- Cheap dev metric, simulated offline on the cached 4B full walks

Question (wassname): can a cheaper dev run rank methods like the full run? Simulated on cached answers and Jev ratings, Qwen3.5-4B, 19 methods incl. random, seed 0; reference = full headline score (100 q, 3 seeds). Source: `slop/reviews/2026-09-29_dev_metric/sim.md` (script `sim.py`, same folder).

| rule | all 100 q | every 5th q | 20 random q, median [5%, 95%] |
|---|---|---|---|
| one dose at 0.67 x breakdown | +0.51 | +0.44 | +0.51 [+0.36, +0.64] |
| one dose at 0.4 x breakdown | +0.86 | +0.78 | +0.72 [+0.49, +0.88] |
| current: walk + best judged dose | +0.99 | +0.81 | +0.82 [+0.69, +0.91] |

(Spearman vs full score; breakdown = first rung the health check flags; one dose per side, score = min over sides of on - off.)

Observations: cutting answers at 128 tokens moves the health-check breakdown rung on 3 of 38 method-sides. At 0.67 x breakdown many methods are already damaged (e.g. directional_ablation dev score -2.46): the judge sees damage well before the health check flags breakdown. 0.4 x is the best of the 5 factors tried (0.67, 0.5, 0.4, 0.3, 0.2). Choosing 20 "informative" questions (largest spread across methods on one half of the methods) ranks the other half worse than every 5th (-0.07, +0.35 vs +0.32, +0.61).

Interpretation (PI/Claude): with 20 questions the ranking ceiling is about 0.8 whatever the dose rule, so question count, not the dose rule, limits dev. The 128-token cap is safe and cuts most generation cost. A one-dose rule saves judge calls (cheap) but not GPU unless breakdown is found with fewer rungs (bisection); 0.4 was picked on the same data, so expect a little optimism. One model, one seed.

-- PI/Claude
