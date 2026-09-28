Same methods, same questions, same judge on Qwen3.5-4B, Qwen3.5-27B and OLMo-2-32B-Instruct (PI/Claude, 2026-09-27).

Reads outputs/bsbench/results/{full,27b-full,olmo-full}/points.json, writes compare.md and compare.png next to this file.
score: min over sides of (on − 1 × off) at each side's Pareto-best dose (the headline score).
on-axis ÷ room: on-axis change at the Pareto-best dose ÷ how far the bare answers could still move toward that side
(8 − bare premise level for +C, the bare level for −C), weaker side; damage is handled by the dose choice and the 1.5 cap.
Seeds: 4B 3 per learned method, 27B 3, OLMo 1 (cross-seed cos of the vectors is 0.99+, so extra seeds add little).

| method | Qwen3.5-4B score | Qwen3.5-4B on-axis ÷ room | Qwen3.5-27B score | Qwen3.5-27B on-axis ÷ room | OLMo-2-32B score | OLMo-2-32B on-axis ÷ room |
|---|---|---|---|---|---|---|
| mean_diff | +0.37 [+0.15, +0.78] | +0.14 [+0.08, +0.24] | +0.91 [+0.60, +1.33] | +0.64 [+0.50, +0.76] | +0.21 [+0.04, +0.50] | +0.27 [+0.21, +0.33] |
| chars | +0.88 [+0.49, +1.23] | +0.35 [+0.22, +0.43] | +0.43 [+0.01, +1.11] | +0.36 [+0.07, +0.45] | +0.13 [-0.04, +0.34] | +0.23 [+0.11, +0.27] |
| vjp_cache | +1.14 [+0.75, +1.56] | +0.40 [+0.30, +0.48] | +0.34 [+0.09, +0.69] | +0.33 [+0.19, +0.47] | -0.20 [-0.38, -0.10] | +0.00 [-0.04, +0.04] |
| vjp_delta | +0.66 [+0.39, +1.14] | +0.24 [+0.18, +0.39] | -0.01 [-0.13, +0.25] | +0.13 [-0.01, +0.25] | -0.05 [-0.16, +0.03] | +0.03 [-0.01, +0.05] |
| vjp_delta-t48 | — | — | +0.21 [-0.02, +0.46] | +0.23 [+0.12, +0.35] | — | — |
| vjp_delta-t47 | — | — | — | — | -0.08 [-0.12, -0.05] | +0.00 [-0.01, +0.03] |
| random | -0.07 [-0.22, +0.13] | +0.01 [-0.03, +0.06] | -0.05 [-0.12, +0.14] | +0.02 [-0.03, +0.13] | -0.10 [-0.15, -0.02] | +0.01 [-0.01, +0.06] |
