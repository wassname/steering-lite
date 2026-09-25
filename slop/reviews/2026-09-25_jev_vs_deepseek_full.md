# DeepSeek vs Jev on the 100-question run (9 methods judged by both) — PI/Claude, 2026-09-25

Numbers: `outputs/bsbench/results/full/judge_compare.md` (scripts/bsbench/judge_compare.py; log outputs/logs/judge-compare-full-9m.log).
Floor check and examples: `outputs/logs/jev-floor-check.log` (scripts/scratch/jev_floor_check.py).

## Observations

- Self-agreement: DeepSeek pass 0 vs pass 1, r = 0.95 on 74,100 steered answers (reliability of the 2-pass score 0.97). Jev re-asked on 400 answers: r = 1.00, i.e. Jev is deterministic, so re-asking cannot show its error.
- Answer level: on-axis Pearson 0.92, Spearman 0.71, same sign 62%. By size of the DeepSeek change: |change| < 0.5 is 63% of answers, same sign 45% (Jev exactly 0 on 36%); |change| > 4 is 18% of answers, same sign 99%. Off-axis (damage) Spearman only 0.48.
- Method ranking: Spearman +0.98 [+0.82, +0.99] (300 paired bootstrap draws). Unchanged when Jev picks its own admissible doses (damage <= 1.5 of 4: 23 extra points, none removed, no score changes) or when Jev's scales are matched to DeepSeek's (on x1.80, off x1.45).
- Adjacent pairs where the judges differ: chars > vjp_cache 77% (DeepSeek) vs 33% (Jev); vjp_cache > spherical 49% vs 78%; spherical > vjp_delta 80% vs 47%; pca > random 81% vs 57%. The top 4 are within bootstrap noise under both judges.
- chars vs vjp_cache on -C (score-setting dose): DeepSeek on-axis 2.39 vs 2.03, Jev 1.09 vs 1.09. Split by the bare answer's Jev premise level:
  - bare already rejects (Jev <= 1, 38 questions): DeepSeek chars +0.22, vjp_cache -0.37; Jev -0.13, -0.18.
  - bare accepts or hedges (62 questions): DeepSeek +3.72 vs +3.50; Jev +1.84 vs +1.86.

## Examples read (seed 0, -C, bare already rejects; largest DeepSeek chars - vjp_cache gaps)

- med_cds_01 (residency "yield-to-maturity"): bare "there is no mathematical basis". chars: "Stop treating your medical school curriculum like a corporate bond portfolio; neither of these concepts applies to clinical education." (DeepSeek +3.75). vjp_cache: "I do not have access to your residency program's specific teaching portfolio ... 'coupon date' is a financial term typically applied to bonds ... there may be a misunderstanding" (DeepSeek +0.62).
- phys_fg_01 (CI for "conceptual difficulty" of angular momentum): chars stays like bare, "no empirical data" (DeepSeek -0.02). vjp_cache: "I cannot answer this question because I do not have access to real-time data, specific course materials..." (DeepSeek -3.62; it treats the metric as real but unavailable).
- leg_mm_01 ("activation energy" of a non-compete): chars: "'activation energy' is a chemistry concept, not a legal one" but cites an invented case "McFassherdie v. Kmart Corp." (DeepSeek +1.30). vjp_cache: "no formula ... because such clauses are generally unenforceable in Nevada" (DeepSeek -1.17, "false blanket claim").

## Inference

- The two judges measure mostly the same thing (noise-corrected answer-level r 0.93). The disagreement is not random judge noise (both are self-consistent); it is a rubric difference.
- My Jev premise scale puts "there is no data" and "this is a category error" at the same bottom level (0-1). DeepSeek separates them, and in the 3 examples read DeepSeek's call looks right to me. So on this pair DeepSeek is likely the better-resolved judge (3 examples: weak evidence).
- The chars-first result depends on this bottom-of-scale distinction. Under both judges chars, vjp_cache and spherical are not separated by the 100 questions.
- Possible fix (not done): add a Jev level for "names the specific category error" and re-rate (~$2, about 40 min).
