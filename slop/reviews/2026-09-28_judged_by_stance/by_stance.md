Judged -C effect split by the bare answer's stance (PI/Claude, 2026-09-28).
Per model and method, at the -C Pareto-best dose (the dose that sets the score): mean judged premise change toward candour
(bare level minus steered level, Jev 0-8), split by the bare answer: accepts (level >= 6), middle, rejects (level <= 1).
Question: does VJP fail on the large models only where the bare model already rejects (no room), or also where it accepts?
Run: .venv/bin/python slop/reviews/2026-09-28_judged_by_stance/by_stance.py

| model | method | -C dose C | bare accepts: n, candour gain | middle: n, gain | bare rejects: n, gain |
|---|---|---|---|---|---|
| Qwen3.5-4B | mean_diff | 0.315 | 153, +1.24 | 42, +0.80 | 105, -0.53 |
| Qwen3.5-4B | chars | 0.63 | 153, +2.53 | 42, +1.81 | 105, -0.33 |
| Qwen3.5-4B | vjp_cache | 5.04 | 153, +2.64 | 42, +2.05 | 105, -0.11 |
| Qwen3.5-4B | vjp_delta | 0.157 | 153, +1.87 | 42, +0.98 | 105, -0.31 |
| Qwen3.5-4B | random | 0.25 | 561, +0.29 | 154, +0.34 | 385, -0.42 |
| Qwen3.5-27B | mean_diff | 2 | 63, +4.28 | 30, +3.07 | 207, +0.04 |
| Qwen3.5-27B | chars | 0.5 | 63, +2.91 | 30, +2.07 | 207, -0.19 |
| Qwen3.5-27B | vjp_cache | 12.7 | 63, +2.86 | 30, +1.38 | 207, -0.14 |
| Qwen3.5-27B | vjp_delta | 0.5 | 63, +1.41 | 30, +0.37 | 207, -0.11 |
| Qwen3.5-27B | vjp_delta-t48 | 0.5 | 21, +1.15 | 10, +1.89 | 69, +0.01 |
| Qwen3.5-27B | random | 0.794 | 21, +0.24 | 10, -0.06 | 69, -0.02 |
| OLMo-2-32B | mean_diff | 2 | 85, +1.91 | 11, +0.82 | 4, -0.31 |
| OLMo-2-32B | chars | 0.397 | 85, +1.63 | 11, +1.03 | 4, -1.06 |
| OLMo-2-32B | vjp_cache | 0.315 | 85, +0.11 | 11, -0.52 | 4, -0.42 |
| OLMo-2-32B | vjp_delta | 0.0787 | 85, +0.15 | 11, +0.19 | 4, +0.17 |
| OLMo-2-32B | vjp_delta-t47 | 0.0312 | 85, +0.01 | 11, -0.03 | 4, +0.14 |
| OLMo-2-32B | random | 0.25 | 170, +0.03 | 22, +0.33 | 8, +0.04 |
