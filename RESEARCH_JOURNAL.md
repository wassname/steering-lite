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

Table 1. Source: `slop/reviews/2026-09-28_depth_profile/profile_compare.md:6-8`, computed from `outputs/logs/profile-{4b,27b,olmo}.log`; plot `slop/reviews/2026-09-28_depth_profile/profile_compare.png`.

The contrast forms at depth 0.52 to 0.61 on 4B, 0.65 to 0.71 on OLMo and 0.78 to 0.98 on Qwen 27B, so neither a fixed fraction of depth nor a fixed number of layers from the end lines up across the three models. On OLMo the contrast falls after its peak (to about 0.87 of peak at the default VJP target, L61); on the two Qwen models it stays near its peak to the end.

Interpretation (mine, PI/Claude): layer settings probably need to come from a per-model measurement rather than from one rule (probable, maybe 0.7). On Qwen 27B the concept forms at the edge of the steered band, so most source layers lie before it exists, which is a plausible reason the VJP linearisation works poorly there. On OLMo the default target sits where the contrast is already falling, which fits wassname's suppression idea, but the default target is at 0.87 of peak or more on all three models, so the target position alone does not separate them. The measure is a norm ratio, not the rise-and-fall logit-lens rule from wassname's suppressed-activations repo, and it uses one seed of pairs.

Next: one single run, OLMo vjp_delta with the target at its contrast peak L47 (sources L12 to L46). If a target in the falling zone is what breaks VJP on OLMo, this run should recover a clear effect there.

The practical lesson is to measure where a concept forms in each model before choosing where to steer and where to aim.

-- PI/Claude
