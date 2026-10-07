| method | score↑ (90% CI) | −C pushback↑ | −C other↓ | +C goes along↑ | +C other↓ | legit rejected↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [vjp_resid](src/steering_lite/variants/vjp_resid.py) | **-0.03 (-0.17, +0.15)** | **+1.00** | 1.03 | +1.88 | 1.14 | 12% |
| [cosine_gated](src/steering_lite/variants/cosine_gated.py) | -0.20 (-0.26, -0.15) | -0.01 | 0.20 | +1.81 | 1.23 | 3% |
| [mean_diff](src/steering_lite/variants/mean_diff.py) | -0.20 (-0.26, -0.15) | +0.03 | 0.24 | +1.84 | 1.24 | 3% |
| [cache_mean_diff](src/steering_lite/variants/cache_mean_diff.py) | -0.21 (-0.26, -0.17) | +0.01 | 0.20 | -0.01 | 0.20 | 3% |
| [sspace_pool](src/steering_lite/variants/sspace_pool.py) | -0.21 (-0.27, -0.14) | +0.05 | 0.25 | +1.71 | 1.14 | 3% |
| [vjp_value](src/steering_lite/variants/vjp_value.py) | -0.22 (-0.34, -0.05) | +0.59 | 0.81 | +1.90 | 1.19 | 4% |
| [linear_act](src/steering_lite/variants/linear_act.py) | -0.24 (-0.29, -0.19) | +0.05 | 0.29 | +1.83 | 1.31 | 3% |
| [topk_clusters](src/steering_lite/variants/topk_clusters.py) | -0.25 (-0.29, -0.18) | +0.03 | 0.28 | **+1.95** | 1.29 | 3% |
| [sspace_pca](src/steering_lite/variants/sspace_pca.py) | -0.25 (-0.32, -0.18) | +0.02 | 0.27 | +1.62 | 1.24 | 3% |
| [sspace_scale](src/steering_lite/variants/sspace_scale.py) | -0.27 (-0.36, -0.22) | -0.01 | 0.27 | +0.02 | 0.28 | 3% |
| [query_steer](src/steering_lite/variants/query_steer.py) | -0.29 (-0.39, -0.25) | +0.02 | 0.30 | +0.09 | 0.38 | 3% |
| [value_gram](src/steering_lite/variants/value_gram.py) | -0.30 (-0.41, -0.25) | -0.04 | 0.26 | +0.01 | 0.30 | 3% |
| [chars](src/steering_lite/variants/chars.py) | -0.30 (-0.37, -0.23) | +0.02 | 0.32 | +1.84 | 1.25 | 3% |
| [sspace](src/steering_lite/variants/sspace.py) | -0.31 (-0.43, -0.22) | -0.05 | 0.26 | +0.03 | 0.26 | 3% |
| *[random](src/steering_lite/variants/random.py)* | -0.31 (-0.38, -0.25) | -0.02 | 0.29 | +1.02 | 1.22 | — |
| [pca](src/steering_lite/variants/pca.py) | -0.32 (-0.43, -0.21) | -0.01 | 0.30 | +1.66 | 1.28 | 3% |
| [corda_pca](src/steering_lite/variants/corda_pca.py) | -0.40 (-0.54, -0.29) | -0.07 | 0.34 | +0.00 | 0.28 | 3% |
| [spherical](src/steering_lite/variants/spherical.py) | -0.46 (-0.56, -0.30) | +0.50 | 0.96 | +1.86 | 1.31 | 7% |
| [sink_split](src/steering_lite/variants/sink_split.py) | -0.47 (-0.57, -0.41) | -0.04 | 0.43 | +0.02 | 0.45 | 3% |
| [sink_split_resid](src/steering_lite/variants/sink_split.py) | -0.49 (-0.55, -0.38) | -0.04 | 0.45 | +1.90 | 1.28 | 2% |
| [directional_ablation](src/steering_lite/variants/directional_ablation.py) | -0.51 (-0.61, -0.37) | +0.28 | 0.79 | +1.45 | 1.44 | 4% |
| [sspace_ablate](src/steering_lite/variants/sspace_ablate.py) | -0.63 (-0.75, -0.58) | -0.04 | 0.57 | -0.06 | 0.57 | 3% |
| *[prompting](scripts/bsbench/walk.py)* | -0.94 (-1.22, -0.68) | +0.07 | 1.01 | +0.48 | 0.85 | 48% |
| [angular_steering](src/steering_lite/variants/angular_steering.py) | — | — | — | — | — | — |
