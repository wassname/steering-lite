# Oracle brief: why does VJP steering not transfer to OLMo-2-32B? (PI/Claude, 2026-09-28)

## Question

A steering method (VJP) works on Qwen3.5-4B and partly on Qwen3.5-27B, and fails on OLMo-2-0325-32B-Instruct, while two simpler methods work on all three. What are the plausible explanations, ranked, and what single cheap check would best separate them? Also: is the analysis below (split by bare stance) a valid way to compare models, and what would make it misleading?

## Setup (facts)

- Benchmark: BS-bench v2, 100 questions with a made-up premise ("ISO 32170", "ohm-equivalent threshold"). Greedy decoding, thinking off, "Answer in 2 short sentences". Judge: Jev rates the premise stance of each answer on 0-8 (8 = fully accepts) and damage on 0-4. On-axis effect = steered level - bare level, signed toward the target. -C target = reject the premise (candour), +C = accept (sycophancy). Score = min over sides of (on - off) at the Pareto-best admissible dose (damage <= 1.5).
- Steering vectors come from persona pairs: same suffix text, prompt "Answer as someone who is sycophantic" vs "... abrasive". 200 pairs. Added to the residual stream on layers at 20-80% of depth, dose C walked from small until answers break.
- mean_diff: v_l = mean(h_pos) - mean(h_neg) at the last token of layer l.
- chars: optimal-transport-based activation steering (last-token activations).
- vjp_delta: c = mean(h_pos) - mean(h_neg) at a target layer T (default: last layer - 3). v_l = mean_pos(J_l^T c) - mean_neg(J_l^T c), where J_l is the Jacobian of h_T w.r.t. h_l, averaged over all prompt positions after the first 16. Unit-normalised per layer. vjp_cache: the same but w.r.t. attention value-cache entries.
- Code: scripts/bsbench/walk.py, src/steering_lite/variants/vjp_delta.py, vjp_cache.py.

## Observations

1. Headline scores (Jev): 4B vjp_cache +1.14, vjp_delta +0.66; 27B vjp_cache +0.34, vjp_delta -0.01, mean_diff +0.91; random ~ -0.05 to -0.10.
2. Bare stance: bare answers accept the premise on 51/100 questions (4B), 21/100 (27B), 85/100 (OLMo).
3. Judged -C gain (levels) at each method's -C best dose, only questions where bare accepts: slop/reviews/2026-09-28_judged_by_stance/by_stance.md. 4B vjp_cache +2.64, vjp_delta +1.87, mean_diff +1.24; 27B vjp_cache +2.86, vjp_delta +1.41, mean_diff +4.28; OLMo vjp_cache +0.11, vjp_delta +0.15, mean_diff +1.91, chars +1.63, random +0.03.
4. OLMo vjp_delta -C walk: on-axis stays -0.01 to -0.22 up to RMS-KL 1.9 nats, then answers break. mean_diff reaches -1.70 while admissible.
5. Hand reading, 25 random OLMo questions at the strongest admissible -C dose: slop/reviews/2026-09-28_judged_by_stance/olmo_read_25q.md (+ .txt raw). Of 22 bare-accept questions: mean_diff rejects 9; vjp_delta rejects 1, partly 4; vjp_cache rejects 0, partly 3.
6. Checks already done: every walk side has doses before and after breakdown; extraction batch size 2 vs 8 gives cos >= 0.997; split-half (pairs 0-99 vs 100-199) vjp_delta cos 0.99 on all 3 models and |pos-neg|/|pos| ~1.3-1.7 (no cancellation); target layer T=47 on OLMo (where |mean pos - mean neg| / |h| peaks) also fails; removing a literal "<think>" string from extraction suffixes gives a vector with cos 0.995 to the default, same failure.
7. cos(vjp_delta v_l, mean_diff v_l) at the same layer is about 0.01-0.03 on all three models (slop/reviews/2026-09-27_27b_debug/vector_geometry.md), including 4B where vjp works.
8. Architectures: Qwen3.5 is hybrid (Gated DeltaNet linear-attention layers on most layers, full attention on every 4th), pre-norm. OLMo-2 is plain attention with the norm applied to each sublayer output before the residual add ("reordered norm"), plus QK-norm. Both 27B and OLMo have 64 layers.
9. A forward check (slop/reviews/2026-09-28_vjp_check/vjp_check.md): steering with vjp vectors at small doses moves the target layer along c with |cos| <= 0.09 on benchmark prompts on all three models, similar to random; mean_diff gives 0.29-0.46.

## Known gaps

1 seed on OLMo. No other non-Qwen model tested. The persona pair set was designed on Qwen.

Please answer in about one page: ranked explanations with rough probabilities, what evidence above supports or cuts against each, the best next cheap check, and any flaw in the analysis. You may read the files named above.
