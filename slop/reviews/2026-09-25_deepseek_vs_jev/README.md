# DeepSeek vs Jev, the last comparison before the switch — PI/Claude, 2026-09-25

On 2026-09-25 the user chose Jev as the only judge: > "jev seems better. visually more consistent and less wiggle. lets usei t from now on! clean up and have jev as one judge"

Frozen copies from outputs/bsbench/results/full/ at commit 7844eb5 (100 questions, 17 methods, seeds 0-2, random 0-10):

- `judge_compare.md`: judge self-agreement (DeepSeek pass 0 vs 1 r=0.94; Jev deterministic), answer-level agreement, rank Spearman 0.97 [0.86, 0.97].
- `judge_frontiers.png`, `judge_scores.png`: frontiers with bootstrap draws and score scatter, Jev rescaled to DeepSeek units.
- `plot_deepseek.png`, `index_deepseek.md`: the DeepSeek results page (vjp-steering reference judge, comparable with its README).
- `plot_jev_rubric_v1.png`: Jev with rubric v1 (premise scale 0-6).
- Reading notes and quoted examples: ../2026-09-25_jev_vs_deepseek_full.md.

To rerun: `git checkout 7844eb5 -- scripts/bsbench` (judge.py = DeepSeek, jev.py, judge_compare.py, judge_plot.py). The DeepSeek and blind caches stay in outputs/bsbench/judgments/{judgments,blind}.jsonl.

Two differences found, and what was done:
1. -C floor: rubric v1 put "there is no data" and "this is a category error" at the same level, so Jev could not see extra candour when the bare answer already rejects the premise. Fixed in rubric v2 (premise scale 0-8, judge.py PREMISE).
2. +C damage: Jev scores fluent, empty agreement as damaged; DeepSeek scores it clean. The vjp-steering confound rubric says "Rate vagueness as severe damage", so Jev follows the reference rule here. Not changed.
