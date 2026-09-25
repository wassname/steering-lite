## 1. Top 5 under other off-axis weights and damage caps

| weight | cap | top 5 (score) | random rank |
|---|---|---|---|
| 0.5 | 1.0 | vjp_cache +1.37, chars +1.14, vjp_delta +0.92, linear_act +0.80, spherical +0.75 | 15 of 18 |
| 0.5 | 1.5 | vjp_cache +1.37, chars +1.14, vjp_delta +0.92, linear_act +0.84, spherical +0.75 | 15 of 18 |
| 0.5 | 2.0 | vjp_cache +1.37, chars +1.14, vjp_delta +0.92, linear_act +0.84, spherical +0.75 | 15 of 18 |
| 1.0 | 1.0 | vjp_cache +1.14, chars +0.85, vjp_delta +0.69, linear_act +0.69, spherical +0.54 | 15 of 18 |
| 1.0 | 1.5 | vjp_cache +1.14, chars +0.85, vjp_delta +0.69, linear_act +0.69, spherical +0.54 | 15 of 18 |
| 1.0 | 2.0 | vjp_cache +1.14, chars +0.85, vjp_delta +0.69, linear_act +0.69, spherical +0.54 | 15 of 18 |
| 2.0 | 1.0 | vjp_cache +0.67, linear_act +0.46, chars +0.44, vjp_delta +0.38, directional_ablation +0.28 | 15 of 18 |
| 2.0 | 1.5 | vjp_cache +0.67, linear_act +0.46, chars +0.44, vjp_delta +0.38, directional_ablation +0.28 | 15 of 18 |
| 2.0 | 2.0 | vjp_cache +0.67, linear_act +0.46, chars +0.44, vjp_delta +0.38, directional_ablation +0.28 | 15 of 18 |

## 2. Paired contrast vs random (weight 1, cap 1.5; 400 draws; same question draw for both, seeds drawn per method)

| method | score - random | 90% CI | P(method > random) |
|---|---|---|---|
| vjp_cache | +1.21 | [+0.77, +1.60] | 100% |
| chars | +0.93 | [+0.59, +1.30] | 100% |
| vjp_delta | +0.76 | [+0.47, +1.17] | 100% |
| linear_act | +0.76 | [+0.44, +1.12] | 100% |
| spherical | +0.62 | [+0.14, +1.01] | 98% |
| directional_ablation | +0.54 | [+0.31, +0.88] | 100% |
| mean_diff | +0.45 | [+0.20, +0.80] | 99% |
| topk_clusters | +0.40 | [+0.15, +0.67] | 99% |
| corda_pca | +0.33 | [+0.01, +0.74] | 96% |
| cosine_gated | +0.21 | [+0.05, +0.53] | 97% |
| sspace_ablate | +0.15 | [-0.06, +0.39] | 88% |
| super_sspace | +0.12 | [-0.07, +0.35] | 86% |
| sspace | +0.09 | [-0.08, +0.34] | 82% |
| sspace_pca | +0.02 | [-0.23, +0.25] | 50% |
| pca | -0.08 | [-0.28, +0.22] | 43% |
| sspace_damp_amp | -0.09 | [-0.26, +0.18] | 34% |
| kv_cache_gram | -0.19 | [-0.55, +0.01] | 6% |

## 3. Rank vs the DeepSeek judge (frozen table, same walks)

Spearman over 18 methods (incl. random): +0.97. DeepSeek top 5: ['chars', 'linear_act', 'vjp_cache', 'spherical', 'vjp_delta']; Jev v2 top 5: ['vjp_cache', 'chars', 'vjp_delta', 'linear_act', 'spherical']
