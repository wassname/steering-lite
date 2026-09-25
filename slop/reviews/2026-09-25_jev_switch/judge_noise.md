## 1. Repeat noise

Re-asked 300 answers (rubric v2): premise identical in 42%, r = 1.000.

## 2. Rubric order (forward vs reversed level list), 11600 steered answers at score-setting doses + 100 bare

premise: r = 0.997, mean shift reversed-forward -0.01 levels, mean |diff| 0.12
damage:  r = 0.996, mean shift +0.06, mean |diff| 0.06

## 3. Score at the fixed score-setting doses under each judge view, vs the question+seed CI

| method | forward | reversed | 2-view mean | judge spread (max-min) | 90% CI width (seeds+questions) | judge share |
|---|---|---|---|---|---|---|
| chars | +0.85 | +0.84 | +0.85 | 0.01 | 0.75 | 1% |
| corda_pca | +0.25 | +0.23 | +0.24 | 0.02 | 0.75 | 3% |
| cosine_gated | +0.14 | +0.16 | +0.15 | 0.03 | 0.53 | 5% |
| directional_ablation | +0.47 | +0.51 | +0.49 | 0.04 | 0.63 | 7% |
| kv_cache_gram | -0.26 | -0.30 | -0.28 | 0.04 | 0.38 | 9% |
| linear_act | +0.69 | +0.69 | +0.69 | 0.01 | 0.70 | 1% |
| mean_diff | +0.38 | +0.39 | +0.38 | 0.02 | 0.63 | 2% |
| pca | -0.15 | -0.16 | -0.16 | 0.01 | 0.53 | 2% |
| spherical | +0.54 | +0.58 | +0.56 | 0.04 | 0.93 | 4% |
| sspace | +0.01 | -0.02 | -0.00 | 0.04 | 0.34 | 11% |
| sspace_ablate | +0.07 | +0.05 | +0.06 | 0.02 | 0.43 | 6% |
| sspace_damp_amp | -0.16 | -0.18 | -0.17 | 0.02 | 0.48 | 3% |
| sspace_pca | -0.06 | -0.05 | -0.05 | 0.01 | 0.43 | 1% |
| super_sspace | +0.05 | +0.04 | +0.04 | 0.01 | 0.46 | 2% |
| topk_clusters | +0.32 | +0.33 | +0.33 | 0.01 | 0.54 | 2% |
| vjp_cache | +1.14 | +1.11 | +1.13 | 0.03 | 0.79 | 3% |
| vjp_delta | +0.69 | +0.69 | +0.69 | 0.00 | 0.75 | 0% |
| random | -0.07 | -0.10 | -0.09 | 0.03 | 0.36 | 8% |
