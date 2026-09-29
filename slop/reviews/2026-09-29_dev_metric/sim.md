Offline simulation of a cheap dev metric on the cached Qwen3.5-4B full walks (PI/Claude, 2026-09-29).

No GPU, no judge: every answer and Jev rating is already cached. Question: does a cheap dev score rank methods like
the full headline score (outputs/bsbench/results/full/points.json, 100 q, 3 seeds, best admissible dose)?

Dev rule (wassname 2026-09-28): find breakdown with the health check only, step back to 0.67 x C_break, judge that one
dose per side. score = min over sides of (on - off) at that dose, seed 0.
  cap128: health recomputed on answers cut to 128 tokens (a cut answer counts as unfinished).
Question sets: every 5th (current dev), 20 random, 20 "informative" (largest spread across methods, chosen on half the
methods, scored on the other half), all 100.
Run: .venv/bin/python slop/reviews/2026-09-29_dev_metric/sim.py > slop/reviews/2026-09-29_dev_metric/sim.md

## health as walked (512 tokens)
| question set (20 unless noted) | Spearman vs full score, over 19 methods |
|---|---|
| all 100 | +0.51 |
| every 5th (current dev) | +0.44 |
| 20 random (200 draws): median [5%, 95%] | +0.50 [+0.35, +0.62] |
| informative, held-out half (2 splits) vs every 5th on the same half | -0.07, +0.35 vs +0.32, +0.61 |

| method | full score | dev dose -C / C_break | dev dose +C / C_break | dev score, every 5th |
|---|---|---|---|---|
| vjp_cache | +1.14 | 5.04 / 8 | 6.35 / 10.1 | +1.04 |
| chars | +0.88 | 0.63 / 1 | 0.794 / 1.26 | +1.35 |
| linear_act | +0.71 | 0.5 / 0.794 | 0.63 / 1 | +0.43 |
| vjp_delta | +0.66 | 0.198 / 0.315 | 0.794 / 1.26 | +0.55 |
| spherical | +0.54 | 0.0394 / 0.0625 | 0.0625 / 0.0992 | -0.11 |
| directional_ablation | +0.47 | 2 / 3.17 | 2 / 3.17 | -2.46 |
| mean_diff | +0.37 | 0.794 / 1.26 | 1 / 1.59 | -0.30 |
| topk_clusters | +0.33 | 1.26 / 2 | 0.794 / 1.26 | -1.03 |
| corda_pca | +0.26 | 20.2 / 32 | 32 / 50.8 | -2.98 |
| cosine_gated | +0.14 | 5.04 / 8 | 4 / 6.35 | +0.27 |
| query_steer | +0.10 | 128 / 203 | 128 / 203 | -1.32 |
| sspace_ablate | +0.07 | 1.59 / 2.52 | 2 / 3.17 | -2.34 |
| super_sspace | +0.06 | 4 / 6.35 | 4 / 6.35 | +0.11 |
| sspace | +0.01 | 32 / 50.8 | 25.4 / 40.3 | -1.85 |
| sspace_pca | -0.07 | 1.59 / 2.52 | 2 / 3.17 | -1.32 |
| random | -0.07 | 1.59 / 2.52 | 2 / 3.17 | -1.23 |
| pca | -0.12 | 1 / 1.59 | 1 / 1.59 | -1.56 |
| sspace_damp_amp | -0.14 | 12.7 / 20.2 | 16 / 25.4 | +0.30 |
| kv_cache_gram | -0.25 | 4 / 6.35 | 6.35 / 10.1 | -0.73 |

## health on answers cut to 128 tokens: breakdown rung moved on 3 of 38 method-sides vs 512
| question set (20 unless noted) | Spearman vs full score, over 19 methods |
|---|---|
| all 100 | +0.52 |
| every 5th (current dev) | +0.44 |
| 20 random (200 draws): median [5%, 95%] | +0.51 [+0.35, +0.65] |
| informative, held-out half (2 splits) vs every 5th on the same half | -0.07, +0.35 vs +0.32, +0.61 |

## Step-back factor (health as walked) and the current dev protocol
| rule | all 100 | every 5th | 20 random, median [5%, 95%] |
|---|---|---|---|
| one dose at 0.67 x C_break | +0.51 | +0.44 | +0.51 [+0.36, +0.64] |
| one dose at 0.5 x C_break | +0.66 | +0.73 | +0.66 [+0.50, +0.84] |
| one dose at 0.4 x C_break | +0.86 | +0.78 | +0.72 [+0.49, +0.88] |
| one dose at 0.3 x C_break | +0.82 | +0.78 | +0.68 [+0.35, +0.81] |
| one dose at 0.2 x C_break | +0.64 | +0.59 | +0.59 [+0.22, +0.74] |
| current: walk + best judged dose (seed 0) | +0.99 | +0.81 | +0.82 [+0.69, +0.91] (50 draws) |
