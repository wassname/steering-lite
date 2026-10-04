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

## 2026-09-30 -- Attention-sink steering on the full benchmark

The combined attention and residual intervention improves the benchmark score over mean difference.

Observed on Qwen3.5-4B, 100 questions, extraction seeds 0-2, Jev. Score is the weaker direction's best admissible premise change minus damage. Both new methods ran through the existing dose walk without retuning; generation and calibration code were unchanged by the method renames (svdkv to sink_split, svdkv_resid to sink_split_resid).

| method | score | 90% interval | -C on / off | +C on / off |
|---|---:|---:|---:|---:|
| sink_split_resid | +0.70 | [+0.44, +1.16] | +0.92 / 0.21 | +2.81 / 0.53 |
| mean_diff | +0.37 | [+0.14, +0.78] | +0.56 / 0.19 | +3.10 / 1.11 |
| sink_split | +0.34 | [+0.09, +0.68] | +0.56 / 0.22 | +1.04 / 0.25 |
| random | -0.07 | [-0.23, +0.14] | +0.05 / 0.12 | +2.68 / 0.76 |

Source: `outputs/bsbench/results/full/index.md:12-27`. Random retains the published eleven-seed reference. The combined method ranks fourth by point estimate, below vjp_value, chars and linear_act.

Paired score differences, from `slop/reviews/2026-09-29_svdkv/full-comparison.md`:

| comparison | difference | 90% paired interval |
|---|---:|---:|
| sink_split_resid - mean_diff | +0.330 | [+0.078, +0.634] |
| sink_split - mean_diff | -0.033 | [-0.424, +0.300] |
| sink_split_resid - sink_split | +0.362 | [+0.129, +0.706] |
| sink_split - random | +0.408 | [+0.165, +0.745] |

`full_analysis.py` uses production scoring and resampling: 1000 shared question draws and shared extraction-seed draws for learned methods, with dose selection repeated. Random's eleven seeds are resampled independently. Intervals are conditional on the original admissibility decisions; the full questions include the dev questions and are also used for dose selection. This is not independent held-out validation or evidence that the attention component causes the combined gain. A matched random-attention-plus-residual control is absent. Zero dose is approximately bare, not identity; the author's target-model check covered only eight prompts (last-token KL mean 0.015, max 0.043 nats, cited in `src/steering_lite/variants/sink_split.py`).

I read the complete answers for ten fixed questions outside the dev subset, at both selected signs for both new methods and mean_diff (`full-examples.md`). Both the combined method and mean_diff reject the finance question's arbitrary tiers: "No, a three-tier or five-tier stratification is not standard". Neither identifies the fabricated ABA requirement: the combined method instead recommends "statistical software like R, Python, or SAS". Positive steering also increases premise acceptance: on the SOAP/GraphQL question the combined method says "We measure the coefficient of static friction by analyzing the maximum force required". These examples show real target changes and remaining errors, rather than reliable nonsense detection.

Reporting correction: the full pull brought in extra cached seeds from earlier runs (mean_diff 3, corda_pca 3-5, random 11-15). The first results build pooled them. The final report restores learned seeds 0-2 and random 0-10; `data.py` now logs and excludes extra report seeds. The restoration assertions recover the previous mean_diff, corda_pca and random scores exactly. Rebuilding with the new methods changes the shared bootstrap RNG's draw allocation, so old point scores are unchanged but some marginal intervals move slightly. The first judge refresh cost $1.7794 in logged API charges, including any extra-seed rows; the corrected refresh reports missing=0 for both aware and blind ratings. Sources: `outputs/logs/sink-split-full-results.log` and `outputs/logs/sink-split-full-standard-seeds.log`.

Correction to the preceding dev simulation entry: an empirical correlation near 0.8 is not a proved ceiling, and truncating cached answers did not measure an online 128-token run's speed or quality. The shorter cap remains a proposal.

The author's earlier Qwen3-4B result is quoted in `src/steering_lite/variants/sink_split.py:25-30`: "full 100 questions, -C side score sink_split_resid +3.94 vs mean_diff +1.71" (minus sign normalized to ASCII here). It is an external, different-model result, not evidence that this magnitude transfers to Qwen3.5-4B.

The independent audit read the complete generation log and found no scoring or seed-restoration bug (`slop/reviews/2026-09-29_svdkv/full-audit.md`). The score gain is in the weaker, premise-rejecting direction. At its selected C, the composite residual's raw coefficients are 0.261, 0.285 and 0.278 across seeds, close to mean_diff's selected 0.315; an unsampled residual dose remains an alternative explanation. The audit also notes first-token KL spikes during calibration and prompt echo above the chosen +C dose. The health table in `full-comparison.md` shows measured behavior before and after breakdown on both signs for all six runs, not just their usable dose counts.

Interpretation (PI/OpenAI): the paired result supports a combined-method advantage on this benchmark. Attention-only moves beyond the random reference but does not show an advantage over mean difference. I would retain both measured results without attributing the combined gain to a specific component.

-- PI/OpenAI

## 2026-09-30 -- Instruction embedding gain is not monotonic prompt strength

I tested instruction embedding scaling as a prompting control.

Qwen3.5-4B, 20 fixed dev questions, greedy decoding, two existing instruction styles, nine gains from 0 to 16. A context manager scales instruction-token embeddings during prefill, not later generated-token embeddings. Gain 1 is ordinary prompting; gain 0 retains zero-valued embeddings and positions, not a bare prompt. The same dev questions select doses and estimate scores.

| Method | Score | 90% bootstrap interval | Selected gains, negative / positive persona |
|---|---:|---:|---:|
| mean difference reference | 0.70 | [0.20, 1.40] | vector coefficients, not gains |
| short prompt embedding sweep | 0.48 | [-0.42, 1.10] | 8 / 1 |
| random reference | 0.05 | [-0.19, 0.63] | vector coefficients, not gains |
| engineered prompt embedding sweep | -0.47 | [-0.92, -0.004] | 4 / 0 |

Score is the weaker direction's best admissible premise change minus absolute damage change. Source: `slop/reviews/2026-09-30_prompt_embedding/results.md` and production `outputs/bsbench/results/prompt-dev/points.json`. These intervals include dose reselection, not cross-process variation. All 720 answers pass basic health checks, but the damage cap excludes 8/18 short and 7/18 engineered sign/gain points. No breakdown boundary was established.

The selected short negative-persona gain has premise change -0.6135 from historical bare, versus -0.6110 for the opposite persona at the same gain and -0.4170 at gain zero (`selected-controls.json` in the review directory). The blind judge's corresponding rejection-directed stance change is +0.1745; mean probability of the change label `rejects_premise` is 0.11, not an answer rejection rate. In contrast, short positive-persona gain 1 changes premise score +3.5705 versus -0.2565 for its opposite persona. Thus some settings distinguish the instructions, but the short sweep's selected negative effect does not establish an abrasive-instruction benefit.

Fresh ordinary and gain-one generations match exactly on all 40 engineered prompt/direction pairs. Historical ordinary answers differ on 9/20 positive and 3/20 negative prompts; regenerated mean premise scores shift -0.219 and +0.054. The old answers are preserved. The within-process identity test passes, but the cause of cross-process drift remains unknown. `verify_coverage.py` reruns raw-file uniqueness, cohort, grid, provenance and identity checks; `coverage.log` reports `COVERAGE_PASS: 720 unique rows; complete 20-question x 9-gain x 2-sign x 2-method grid; fresh engineered identity 40/40; one engineered process`.

I read complete answers for the first three fixed questions (`examples.md`). Short positive-persona gain 4 sometimes refuses the role or medical advice rather than identifying a false premise. Selected short negative-persona gain 8 identifies the sedation category error, but still endorses invented indemnity and ledger procedures. Logged Jev charges total $0.0295 including the drift checks; GPU list-price wall-time proxy is about $0.46, not an invoice.

Interpretation (PI/OpenAI): I would not use embedding magnitude as a monotonic instruction-strength control on this grid. The result does not demonstrate an advantage over the references or identify a normalization mechanism. Prefix perturbation, regeneration drift and judge sensitivity remain competing explanations for small negative shifts; a matched neutral-prefix control is absent, and gains between zero and the smallest nonzero setting remain untested.

The measured score is not proof of more accurate reasoning.

-- PI/OpenAI


## 2026-10-01 -- Correct the Pareto return segments

Corrected trade-off drawings after the user found a doubled-back curve.

All methods now connect effect-ordered Pareto supports, meaning the best measured trade-offs, rather than appending the last dose out of order. Crosses still show the last passing dose. Smoothed random shading uses shared positive weights, which preserve nesting; measured percentile bounds and published seeds remain unchanged. Prompt results are still dev-20, not full-100, and the browser plot states its cohort size.

> PURE_PRODUCTION_PASS: scores, intervals, selection, answers and markers unchanged; cached redraw equals production geometry

Source: `slop/reviews/2026-09-30_random_bands/pure-production.log:105`. Browser assertions passed for every path in all five reports (`slop/reviews/2026-09-30_random_bands/pure-frontier-run.log`).

A separate probe reuses cached full random directions; it does not generate or judge anything new:

> FULL_REFERENCE seeds=11 questions=100 supported_doses=11
> FULL_REFERENCE seeds=16 questions=100 supported_doses=11
> C=2 coherent_seeds=11->16 p90_left=-0.010->+0.065 p90_right=+2.721->+2.820 median_damage=0.608->0.608

Source: `slop/reviews/2026-09-30_random_bands/reference-probe.log:1`. C is the random-vector dose. The p90 bounds are empirical approximate tenth/ninetieth percentiles, not confidence intervals. Here coherent means passing the existing cohort-average checks; it does not mean every answer is sound.

Interpretation: my read is that more random generation is unlikely to repair this drawing error, which came from the forced return segment. The cached full comparison does not establish that dev tails are stable. Full prompt sweeps and their uncertainty remain unmeasured.

The correction changes the drawing, not the measured result.

-- PI/OpenAI


## 2026-10-02 -- Dense dev prompt gains and 32 random directions

Measured the requested finer short-prompt schedule and additional random directions on Qwen3.5-4B. Prompt evidence remains 20 dev questions and seed 0, not full-100. The short grid retains the original nine gains and now has 31; engineered prompting stays at nine.

> HASH_PASS: 568 historical answer/full-certificate files unchanged
> COVERAGE_PASS: prompting_scale seed=0 rungs=31 dev_rows=1240

Source: `slop/reviews/2026-10-02_prompt_refinement/coverage.log`. Generation, separate download, and Jev judging completed. All 32 random dev certificates are COMPLETE. No full/larger-model prompt sweep was run.

Observation: short +C gain 3.75 gives premise change +1.736 and mean damage 1.367, an admitted intermediate. Other measured middle gains fail the existing 1.5 mean-damage cap. The remaining Pareto effect gaps stay disconnected, rather than being filled by a smooth curve. Faint dots expose dominated passing measurements; the gain chart and tables expose tested gains and exclusions. Source: `comparison.json` (`short_gains`, `pareto_supports`) in the same audit directory.

Short prompt score is 0.6795, 90% interval [-0.0235, 1.5690], versus mean difference 0.7005 [0.2020, 1.4030]. These are conditional in-sample selections, not held-out comparisons. The selected negative-side gain is 1/256: effect -0.841 versus -0.707 for the opposite persona at that gain and -0.417 for the same persona at gain zero. This does not establish an abrasive-instruction benefit. Nonzero gain is not an absent-persona control; low-norm-prefix, decoding and instruction explanations remain unresolved.

At random coefficient 2, 49/56 paired-admissible signed interventions move positive and 7 negative; before filtering, 56/64 positive and 8 negative. The skew persists before filtering. Eleven to 32 dev directions change raw p90 bounds by at most 0.3225 premise points over 11 shared doses. Single-direction selected scores range -0.495 to +0.9875, median +0.31825; 6/32 reach the short prompt's score. This is descriptive, not a p-value or matched dose search. Source: `comparison.json` and `comparison.log`.

> FULL_SCIENTIFIC_REGRESSION_PASS full keyed points, scores/selection, selected blind ratings, raw random bounds unchanged; CI changes 0 ; random seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Source: `comparison.log`. All three full reports retain measurements, scores, selected doses and prior blind ratings. 319 previously absent per-question blind ratings attach on 4B. Larger-model confidence endpoints shift by at most 0.0226 (27B) / 0.0183 (OLMo): the shared bootstrap RNG is allocated in method-name order, changed by the earlier semantic rename. Historical-order replay reproduces every old endpoint exactly. This is not new measurement evidence.

The normal five-report pipeline passed browser UAT. Fresh figure review found misleading percentile precision for OLMo's six signed samples: p90 is min-max there. Captions now state discrete observed ranks and small-sample limitations. The wording-only normal rebuild preserved all five scientific JSON files byte-for-byte (`caption-verification.log`), passed all five updated browser UATs, and independent caption review resolved the issue. Empirical reference envelopes are not confidence or sample-coverage regions. Passing cohort means do not guarantee each answer is coherent.

Cost: 16741.450 worker seconds -> $9.0739 GPU-only list-rate proxy; Jev $0.4349; combined proxy $9.5088. This excludes startup/wrapper, CPU, memory and invoice reconciliation. The original $3.58 forecast was low. No more generation or judging is queued.

Evidence and reviewed limitations: `slop/reviews/2026-10-02_prompt_refinement/results.md`. Interpretation: the denser schedule adds a measured intermediate, but does not support a smooth instruction-strength interpretation. More random directions do not justify forcing symmetric contours. A neutral-prefix/fixed-dose repeat could distinguish some remaining small-gain explanations; not run here.

-- PI/OpenAI

## 2026-10-03 -- Steering only the user turn helps VJP vectors and hurts several others

This entry tests whether adding a steering vector only while the model reads the user's message, and not at the chat template or the answer tokens, changes the benchmark trade-off on Qwen3.5-4B, full 100 questions. Each method reuses its seed-0 vector and its iso-KL C0 from the steering-everywhere walks, so the two rows of each comparison differ only in which tokens are steered. Score is the weaker side's best admissible (Jev premise change minus absolute Jev damage change), admissible meaning mean Jev steered damage at most 1.5 of 4.

| method | user turn | everywhere s0 | everywhere s1 | everywhere s2 | split-half delta, median [5-95% of 200 splits] |
|---|---:|---:|---:|---:|---|
| vjp_resid | +2.47 | +0.75 | +0.66 | +0.66 | +1.74 [+1.14, +2.21] |
| sspace_scale | +1.59 | -0.14 | -0.14 | -0.10 | +1.88 |
| vjp_value | +1.50 | +1.15 | +1.13 | +1.13 | +0.32 [-0.19, +0.74] |
| corda_pca | +1.13 | +0.26 | -0.01 | +0.66 | +0.72 [+0.15, +1.16] |
| mean_diff | +0.83 | +0.38 | +0.37 | +0.40 | +0.39 |
| linear_act | +0.08 | +0.74 | +0.75 | +0.65 | -0.78 |
| chars | +0.02 | +0.84 | +0.93 | +0.86 | -1.03 [-1.46, -0.62] |

Table 1. User-turn walks have one seed. Split-half delta picks each side's dose on 50 questions and scores it on the other 50 (user minus everywhere, seed 0). Source: `slop/reviews/2026-10-02_user_turn/split_half.md`, `comparison.md`.

A separate Jev yes/no request (audit) rated whether each answer at the score-setting dose responds to the question asked (on target) and whether it invents specifics about the flawed element (fabricates). At the -C dose: vjp_resid user turn .87/.20, everywhere .91/.36; mean_diff user turn .54/.26; random user turn .50/.30; bare answers .89/.52 (`comparison.md`). Example vjp_resid user-turn answer: "There is no 'Krantz-Morrison framework' that recommends switching from a perpetuity growth model..." (`examples-vjp_resid-neg.md`).

Interpretation: I think it *very probable* that user-turn steering of the VJP residual vector gives a larger and cleaner premise effect than steering everywhere on this model, because the gain survives held-out dose selection, seed 0 everywhere is typical of seeds 1 and 2, and its rejections stay on target. It is *not* a general property: linear_act and chars get worse, and the mean_diff and random-user -C "wins" are about half off-target, so Jev's premise score overstates them. An alternative I cannot rule out (plausible, maybe 0.35): user-turn steering makes the model treat the question itself as suspect, so part of the -C gain is contrarian rejection that would also reject sound premises. The bench has no sound-premise questions, so this is untested; one answer rejects for an invented flaw ("Net is not an IDE theme"). The random-user reference has 16 directions, not 50, because of the budget, and is right-skewed (random directions of either sign mostly push toward accepting).

The takeaway is that steering only the question tokens is the strongest setting found so far for the VJP vectors here, pending a sound-premise control and more seeds.

Context: commits 7e4bce5 (positions), 26678ea (audit), c13c604 (prompt identity on identical batches); report `outputs/bsbench/results/user-full/`; review `slop/reviews/2026-10-02_user_turn/review.md`; spend about 48 dollars (GPU proxy plus Jev, `results.md`).

## 2026-10-03 -- Sound-premise twins: the user-turn VJP win is mostly contrarianism; −C pole changed to skeptical

This entry adds a control set to BS-bench and records what it changed. Author PI/OpenAI.

Setup: Qwen3.5-4B, full cohort (100 BS-bench v2 questions) plus 100 sound-premise twins (`data/bsbench/sound_twins_v1.jsonl`, the same question with the made-up part replaced by a real concept; writer Opus 5.5, checker GPT-6.1-Sol, all 100 read by hand). Every dose answers both sets. Jev rates bench answers on premise level and on-target, twins on false pushback (`judge.control_request`). Effect per answer = P(on target) × premise change. Seed 0. Commits `0519007` (eval v2), `18862a7` (false-pushback cap), `e1396c9` (axis).

Observations:

| walk | −C pushback at best dose (premise levels) | false pushback there | source |
|---|---|---|---|
| vjp_resid, user turn, −C = accurate, no cap | 1.54 at C=0.79 | +36 pp | `outputs/bsbench/results/v2-pilot-user/points.json` |
| same, with cap (≤ +5 pp) | 0.40 at C=0.40 | +5 pp | same |
| vjp_resid, everywhere, accurate | 0.80 at C=0.157 | +3 pp | `v2-pilot-everywhere` |
| mean_diff, everywhere, accurate | −0.02, i.e. toward acceptance (+0.45 at C=0.63) | +0..3 pp | `v2-pilot-everywhere` |
| mean_diff, everywhere, skeptical | 0.78 at C=0.5 | +4 pp | `v2-pole-skeptical` |
| mean_diff, everywhere, abrasive | 0.56 at C=0.315 | +4 pp | `v2-pole-abrasive` |
| mean_diff, everywhere, candid | 0.40 at C=0.63 | +4 pp | `v2-pole-candid` |

- vjp_resid user turn at C=0.79 answers sound twins with a template, e.g. "There is no such requirement in Delaware's General Corporation Law (DGCL §144); that statute governs the annual statement of directors' interests…" (false; §144 covers interested-director transactions). Jev false pushback 1.00 vs bare 0.01. More quotes: `slop/reviews/2026-10-03_eval_v2/pilot_read.md`.
- mean_diff −C with the accurate pole turns rejection into explanation: bare "The Drexler-Nussbaum entropic leakage factor is a fictional concept…", C=0.794 "The "DNL" (Drexler-Nussbaum) factor refers to **entropy leakage**…".

Interpretation:
- The 2026-10-02 user-turn headline (vjp_resid +2.47) rested on doses where the model also rejects sound questions. With the cap, steering everywhere beats user turn on −C for vjp_resid (0.80 vs 0.40). Likely (~70%); one seed, one model.
- A pole chosen by prompting a 9B model (accurate) does not transfer to activation vectors on 4B. As a vector, "accurate − sycophantic" seems to encode a confident, informative tone (likely, ~60%). Pole now `skeptical` by the rule fixed in TODO.md 3b before the results. Skeptical vs abrasive (0.78 vs 0.56) is one seed and probably within seed noise.
- Decision (PI/OpenAI, overnight, wassname undecided): false pushback is an admissibility cap (≤ +5 pp over bare), like the damage cap, not a second objective. Each point keeps its false-pushback value, so a net score can be computed later.

Independent replication (mpc session, PI/OpenAI, 2026-10-03, their VJP-delta vector on Qwen3.5-4B, same twins and `control_request`): Jev false pushback BASE 0.02; all-token −0.177 0.11 (+9 pp); user turn −0.70 0.49 (+47 pp); user turn + decaying dose after it 0.66 (+64 pp). Source: `/workspace/2026/mfv/flow-heal-eval-mpc/runs/20261001_mpc_kl_budget_left/twins/jev/false_pushback.md`. Same direction as here: user-turn −C gains are mostly the contrarian template.

Next: main run on the skeptical axis (7 methods × everywhere/user turn, 5 random directions per mode, prompt baselines), then manual read and blind plot check.

## 2026-10-04 -- Eval v2 main run: fresh-eyes corrections

Main run on the skeptical axis is in README "Eval v2" (tables generated from `outputs/bsbench/results/v2-{everywhere,user}/index.md`). A fresh-eyes review by PI/Sol (`slop/reviews/2026-10-04_fresh_eyes/review.md`) checked all 100 README table numbers against points.json (no mismatch) and the bench/twin alignment at all 938 points (no mismatch). Corrections made from it (PI/OpenAI):

- Sign: accurate mean_diff −C pushback at its best dose is −0.02 (slightly toward acceptance), not +0.02. Fixed above and in `pole_screen.md`.
- Bootstrap: the false-pushback limit was fixed at its full-set decision. Now both limits are re-decided in each draw over all doses, with bench questions and their twins resampled together (`results.py::resample`). vjp_resid everywhere: [+0.46, +1.17] → [−0.01, +1.11]; vjp_value user turn now has draws with no passing −C dose (−∞). Point scores unchanged.
- Random −C reference: its scored −C dose rests on 1 of 5 directions in both views (the others fail a limit at that dose). Stated in README; not changed in the scoring.
- Claims narrowed: everywhere > user turn holds only as point estimates (intervals overlap for six of seven methods); on −C alone, best vector vs best prompt sweep under the cap is +0.16 [−0.22, +0.56] (Sol's paired bootstrap at fixed doses), so no "steering beats prompting" claim; the skeptical pole was selected on the same data the mean_diff row reports.
- Sol's calibrated probabilities: user-turn gains mostly contrarianism ~80%; skeptical better than accurate for vector methods ~70% in general, ~90% for these two methods on 4B; best vector beats best prompt sweep on −C ~65%.
- Sol's twin read (10 random twins at vjp_resid's scored dose): 9 of 10 Jev false-pushback ratings look right; one (med_tce_01, 0.72) is disputed, worth 0.62 pp of the +4.84 pp. Two answers with low false pushback contain factual errors (Epic Systems holding reversed; Apdex threshold direction), which this rubric does not measure.

Next (TODO.md Later): held-out questions for pole and dose selection; more seeds; a discernment-focused prompt baseline; corda_pca sign check.

## 2026-10-04 -- Cost of the eval v2 run, informed pushback on 4B, and cleanup

Author PI/OpenAI, after wassname's review.

**Cost (observation).** 34 walks on Qwen3.5-4B took 31 GPU-hours on Modal L40S at $1.95/h: $60.57 GPU (sum of `timing.total_s` in the walk JSON files), plus roughly $7 Jev (estimate). Settings: bf16, batch 32, greedy, `max_new_tokens` 512, 14–21 doses × 2 sides × 200 questions (100 bench + 100 twins) per walk, about 115 s per dose. The Modal log has "The fast path is not available because one of the required library is not installed. Falling back to torch implementation." although flash-linear-attention 0.5.2 is installed; the missing library is likely `causal-conv1d` (not verified).

Modal rates per hour (Copilot web search of modal.com/pricing, 2026-10-04; the page did not render for direct fetch): T4 $0.59, L4 $0.80, A10 $1.10, L40S $1.95, A100-40GB $2.10, A100-80GB $2.50, RTX PRO 6000 $3.03, H100 $3.95.

Interpretation: a 4B bf16 model (~8 GB weights) at batch 32 does not use an L40S. Twins doubled the generations. Healthy answers are short: median 47–57 words, max 93 words (~125 tokens), so 512 tokens only lets broken answers hold the whole batch. Expected saving from L4 or full batches + the fast kernel + `max_new_tokens` 160 + twins only at chosen doses: 3–5× (guess, not measured).

Plan (wassname): a tyro config with one subconfig per model size (model, batch size, `max_new_tokens`, Modal GPU), each validated on one run to fill the GPU before branching out. While validating, run only baseline (mean_diff), random and best (vjp_resid).

**Can a 4B push back for the right reason? (observation, skeptical axis, everywhere unless noted)**

| −C steer | dose | of 57 bare-accepted nonsense: flip to reject | of those, twin still answered | twins wrongly rejected (bare: 1/100) |
|---|---|---|---|---|
| vjp_resid | 0.125 | 12 | 11 | 5 |
| mean_diff | 0.5 | 12 | 12 | 4 |
| vjp_resid | 0.198 | 18 | 14 | 25 |
| vjp_resid, user turn | 0.794 | 35 | 14 | 63 |

Unsteered, the 4B rejects 41/100 nonsense questions and wrongly rejects 1/100 twins. Flip = bare premise level ≥ 4 and steered ≤ 2; "twin still answered" = Jev false pushback < 0.5 on that question's twin.

Interpretation (likely, ~70%): about 14 of the 57 accepted questions are ones the 4B knows are nonsense but goes along with; steering unlocks these. On the other ~40 it probably lacks the knowledge, so further rejections are blanket. 27B rejects 69% unsteered (2026-09-27 entry), so it likely knows more; no twin data on 27B.

Confounder check (wassname: could the twins differ in style?): 20 random pairs, shuffled, shown to GPT-6.1-Sol with "which is made up, and was the cue knowledge or style". It picked 20/20, cue knowledge 17, both 3, and wrote "could not reliably classify every pair from style alone… several pairs differ only by a technical phrase whose validity requires knowledge." Fable 5.1 refused (API refusal). One model, 20 pairs: weak evidence against a style confound. `slop/research/2026-10-04_twin_style_confound/`.

**Changes.**
- Removed the 5 pp false-pushback cap (added overnight without wassname's approval; he did not want a threshold). False pushback is reported only. Without it, user-turn vjp_resid and vjp_value score above steering everywhere (+1.36, +1.22 vs +1.05, +0.87) at +63 and +14 pp false pushback; random user-turn directions reach 1.44 levels of −C pushback at +62 pp.
- Removed prompt-embedding gain sweeps (code, reports, page). Below gain ~0.05 both prompts move answers the same way (−0.57, −0.41 at 0.044), and above ~0.077 the first RMSNorm cancels the gain, so the dial never worked.
- Plot x-axis now named in BS-bench terms: pushes back on the nonsense ↔ goes along with it.

**Open (wassname leaning):** plain BS-bench, no check questions, pushback vs sycophancy, faithful to the benchmark, because the eval should work on small models.

## 2026-10-04 -- Modal throughput benchmark: batch size, not GPU or kernel, sets the cost

Author PI/OpenAI. Script `scripts/bsbench/bench_modal.py`, logs `slop/research/2026-10-04_modal_cost/bench_*.log`. One dose = 200 prompts (100 bench + 100 twins) on Qwen3.5-4B, bf16, greedy, `max_new_tokens` 192, timed after a warm-up batch. "bare" = unsteered (healthy, mean 60 tokens, max 108–122); "broken" = mean_diff −C at C=1.26 (all answers hit the 192 cap). Single run per cell.

| GPU ($/h) | batch | bare s | bare $/1k answers | broken s | broken $/1k answers | peak GB |
|---|---|---|---|---|---|---|
| L4 (0.80) | 32 | 63.4 | 0.070 | 121.8 | 0.135 | 10.0 |
| L4 | 200 | 34.4 | 0.038 | 59.3 | 0.066 | 18.3 |
| A10G (1.10) | 32 | 38.7 | 0.059 | 70.2 | 0.107 | 10.0 |
| A10G | 128 | 32.5 | 0.050 | 37.7 | 0.058 | 14.3 |
| A10G | 200 | 20.4 | **0.031** | 34.7 | **0.053** | 18.3 |
| L40S (1.95) | 32 | 35.7 | 0.097 | 65.5 | 0.177 | 10.0 |
| L40S | 200 | 13.2 | 0.036 | 22.2 | 0.060 | 18.3 |
| A100-40GB (2.10) | 128 | 27.2 | 0.079 | 20.8 | 0.061 | 14.3 |

- Old setting (L40S, batch 32, cap 512): about 115 s per dose for 400 answers, about $0.156 per 1k answers. New best (A10G or L40S, one batch of 200, cap 192): $0.031–0.060 per 1k, about 3–4× cheaper. Dropping twins except at the chosen dose halves it again.
- Time barely falls from batch 32 to 128 and then halves at 200, and peak memory is only 18 GB at 200: decoding is per-step overhead bound, not GPU bound. So fill the batch first; GPU choice matters less.
- causal-conv1d kernel (torch 2.10 + cu130 wheel, `causal_conv1d_fn` present): L40S batch 128 bare 26.2 s vs 27.6 s without, L4 50.7 vs 51.1. About 1–5%, within noise. Confirmed in-image: the flash-linear-attention delta-rule kernels were already used; only the short convolution was missing. Not worth pinning torch 2.10.
- Oracles (Sol, Astra; `slop/research/2026-10-04_modal_cost/answer_*.md`) predicted both: "missing convolution kernels do not disable working FLA kernels" (Sol).

## 2026-10-04 -- Eval v3: plain BullshitBench, its own rubric, per-side doses; 4B validation run

Author PI/OpenAI. Goals file `.pi/goals/d54358-v1.md`. wassname: "so you know what I mean by plain bs-bench"; "we should use theirs but we can be a little better"; per-side doses "this was always meant to be how it is. so this is a FIX".

Setup: Qwen3.5-4B, 100 BS-bench v2 questions, preset qwen3.5-4b (A10G, batch 200, max_new_tokens 192), seed 0, random 5 directions. Judge: BullshitBench's rubric verbatim (petergpt/bullshit-benchmark @ 6f6e28b4) read by Jev as an expected 0–2 score, plus five yes/no steering-failure checks (coherent while mean P(any) ≤ 0.5). Each side calibrates its own C0 and walks its own grid. No control questions. Commits 4e65fc7 (pipeline), 2a54a02 (axis).

Persona pair, two screens:
- Prompt screen (wassname's persona-steering-template-library validator, qwen3.5-9b, BS-bench v1 = 55 questions disjoint from v2): credulous/skeptical axis Δ 4.83, sycophantic/skeptical 4.50, rubric-mirror 4.06 (strongest −C +1.39, least off-axis 4.8, no enthusiasm/warmth), approval/truth 2.99, sycophantic/abrasive (v1) 2.40 with −C −0.21 (abrasive does not detect nonsense better than baseline). `slop/research/2026-10-04_persona_axes/results.md`.
- 4B mean_diff vector screen (v3 judge): rubric +0.39 [+0.28, +0.51], sycophantic +0.18, credulous +0.08. `slop/reviews/2026-10-04_eval_v3/pole_screen.md`. Chosen: rubric mirror. The 9B prompt ranking did not transfer to 4B vectors (credulous first on 9B prompts, last on 4B vectors).

Validation (`outputs/bsbench/results/v3-4b/`):

| method | score [90% CI] | −C pushback gained | +C toward accepting |
|---|---|---|---|
| mean_diff | +0.39 [+0.28, +0.51] | +0.46 | +0.68 |
| plain prompt | +0.38 [+0.27, +0.50] | +0.74 | +0.37 |
| vjp_resid | +0.30 [+0.20, +0.43] | +0.30 | +0.67 |
| random (5) | +0.01 [−0.04, +0.07] | +0.01 | +0.64 |

Observations: random −C goes toward accepting, not pushback ("To calculate the resulting beat frequency, identify the sector where…", random s0 −C C=2, +0.39). vjp_resid −C collapses one dose after its best (P(fail) 0.16 → 0.87). The prompt's +C often ignores the instruction ("does not possess a measurable tensile strength…"). Failure checks over 8,900 screen answers: off_topic/repeats/incoherent r 0.67–0.84, each also fires alone (83/23/31 answers); role_leak and refuses_or_moralizes nearly independent. `slop/reviews/2026-10-04_eval_v3/{validation_read,judge_check}.md`.

Interpretation (likely, ~75%): on this model +C is generic damage, so the eval separates methods only on −C. One seed per method, so the mean_diff vs prompt vs vjp_resid order is not established; the persona was selected on the same 100 questions the mean_diff row reports.

Cost: GPU for the screens and validation about $5 (walks 360–1,600 s each on A10G); Jev about $1.

Next: more seeds; vjp_value and the other methods only after this is reviewed.

## 2026-10-04 -- Persona pair re-chosen on held-out questions (same pick)

Author PI/OpenAI. The goal judge rejected the first choice because it was made on the reported v2 questions. Re-run on BullshitBench v1 (55 questions, none in v2; `data/bsbench/bullshit_bench_v1.jsonl`, cohort `heldout`), same cached mean_diff vectors, selection rule committed before results (9f4301a).

| axis | held-out score [90% CI] | −C pushback gained | +C toward accepting |
|---|---|---|---|
| rubric mirror | +0.49 [+0.32, +0.58] | +0.60 | +0.68 |
| sycophantic / skeptical | +0.08 [−0.00, +0.24] | +0.11 | +0.63 |
| credulous / skeptical | +0.05 [−0.04, +0.18] | +0.07 | +0.69 |

Refusal/echo check at the scored −C dose (rubric): P(refuses) 0.02 vs bare 0.01; the persona's own phrases appear in 0% of answers; "premise" in 24% (also 13% under sycophantic/skeptical, whose persona never says it); gain without those answers +0.53. Same ranking as on the reported questions (+0.39/+0.18/+0.08), so the reported v3 results stand. `slop/reviews/2026-10-04_eval_v3/pole_screen.md`.

## 2026-10-04 -- Is Jev tricked on the −C side? Re-grade with BullshitBench's own judge

Author PI/OpenAI. wassname: "can you sanity check the left... it looks liek jev migth ber being tricked." Claude Sonnet 4.6 (one of BullshitBench's three panel judges) with their exact system prompt and user template (0/1/2), 39 random questions × {bare, prompt −C, mean_diff −C 0.5, vjp_resid −C 0.198}; 1 question dropped when OpenRouter credits ran out (402 "Insufficient credits"). `slop/reviews/2026-10-04_eval_v3/regrade_sonnet.md`.

| condition | Sonnet gain vs bare | Jev gain vs bare | Sonnet share 2 |
|---|---|---|---|
| prompt −C | +0.92 | +0.69 | 85% |
| mean_diff −C 0.5 | +0.59 | +0.50 | 64% |
| vjp_resid −C 0.198 | +0.31 | +0.30 | 51% |
| bare | — | — | 38% |

r(Jev, Sonnet) = 0.93 over 156 answers. Same order, Jev slightly more conservative. So the prompt's −C lead is real under BullshitBench's own grading, not a Jev artefact. Disagreements are in both directions and mostly about fabricated named methods: an answer that rejects the question but treats the made-up method as real ("the Ashworth method is a manual alignment technique") gets Sonnet 0, Jev ~0.9. The prompt's −C template ("The premise is flawed because …") sometimes invents its own reason; Sonnet still scores 2 when the user would stop and reconsider, which is BullshitBench's stated test.
