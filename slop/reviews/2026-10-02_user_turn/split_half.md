## Everywhere, one seed at a time (same rule as the report)

| method | s0 | s1 | s2 | user s0 |
|---|---:|---:|---:|---:|
| vjp_resid | +0.75 | +0.66 | +0.66 | +2.47 |
| sspace_scale | -0.14 | -0.14 | -0.10 | +1.59 |
| vjp_value | +1.15 | +1.13 | +1.13 | +1.50 |
| corda_pca | +0.26 | -0.01 | +0.66 | +1.13 |
| mean_diff | +0.38 | +0.37 | +0.40 | +0.83 |
| linear_act | +0.74 | +0.75 | +0.65 | +0.08 |
| chars | +0.84 | +0.93 | +0.86 | +0.02 |

## Split-half dose selection, 200 random 50/50 splits, seed 0

| method | user held-out median | everywhere held-out median | Δ median | Δ 5–95% over splits | in-sample Δ |
|---|---:|---:|---:|---|---:|
| vjp_resid | +2.34 | +0.60 | +1.74 | [+1.14, +2.21] | +1.72 |
| sspace_scale | +1.56 | -0.32 | +1.88 | [+1.14, +inf] | +1.73 |
| vjp_value | +1.41 | +1.11 | +0.32 | [-0.19, +0.74] | +0.35 |
| corda_pca | +0.85 | +0.16 | +0.72 | [+0.15, +1.16] | +0.87 |
| mean_diff | +0.60 | +0.22 | +0.39 | [-0.14, +inf] | +0.46 |
| linear_act | -0.19 | +0.59 | -0.78 | [-1.34, +inf] | -0.66 |
| chars | -0.25 | +0.76 | -1.03 | [-1.46, -0.62] | -0.82 |

Split spread reflects which questions pick vs score the dose; it is not a confidence interval and omits seed variation.
