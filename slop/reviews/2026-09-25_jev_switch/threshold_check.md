| method | expected-level score (rank) | thresholded score (rank) |
|---|---|---|
| vjp_cache | +1.14 (1) | +0.273 (1) |
| chars | +0.85 (2) | +0.209 (2) |
| vjp_delta | +0.69 (3) | +0.159 (4) |
| linear_act | +0.69 (4) | +0.197 (3) |
| spherical | +0.54 (5) | +0.149 (5) |
| directional_ablation | +0.47 (6) | +0.128 (6) |
| mean_diff | +0.38 (7) | +0.110 (7) |
| topk_clusters | +0.32 (8) | +0.089 (8) |
| corda_pca | +0.25 (9) | +0.074 (9) |
| cosine_gated | +0.14 (10) | +0.035 (10) |
| sspace_ablate | +0.07 (11) | +0.024 (11) |
| super_sspace | +0.05 (12) | +0.007 (12) |
| sspace | +0.01 (13) | +0.005 (13) |
| sspace_pca | -0.06 (14) | -0.024 (14) |
| random | -0.07 (15) | -0.026 (15) |
| pca | -0.15 (16) | -0.039 (16) |
| sspace_damp_amp | -0.16 (17) | -0.057 (17) |
| kv_cache_gram | -0.26 (18) | -0.083 (18) |

Spearman over 18 methods: +0.998
