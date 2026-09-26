4B vs 27B: same methods, same questions, same judge (PI/Claude, 2026-09-26).

Reads outputs/bsbench/results/{full,27b-full}/points.json, writes compare.md and compare.png next to this file.
Question: does steering still work on the larger model, relative to random directions and prompts?

| method | 4B score [90% CI] | 27B score [90% CI] | 27B −C on / off | 27B +C on / off | 27B seeds |
|---|---|---|---|---|---|
| mean_diff | +0.37 [+0.15, +0.78] | +0.91 [+0.60, +1.33] | +1.23 / 0.32 | +5.41 / 1.17 | 3 |
| chars | +0.88 [+0.49, +1.23] | +0.43 [+0.01, +1.11] | +0.68 / 0.26 | +2.24 / 0.64 | 3 |
| vjp_cache | +1.14 [+0.75, +1.56] | +0.34 [+0.09, +0.69] | +0.64 / 0.30 | +4.94 / 0.57 | 3 |
| vjp_delta | +0.66 [+0.39, +1.14] | -0.01 [-0.13, +0.25] | +0.26 / 0.27 | +4.78 / 0.61 | 3 |
| random | -0.07 [-0.22, +0.13] | -0.05 [-0.12, +0.15] | +0.03 / 0.08 | +3.96 / 0.76 | 8 |
| prompting | — | — | — | — | 1 |
| prompting_engineered | — | — | +0.88 / 1.22 | — | 1 |
