# Results (full)

Score = min over ±C of (on-axis − 1 × off-axis) at each side's best admissible dose. CI: 1000 hierarchical bootstrap draws (seeds with replacement, then questions with replacement), dose selection redone in each; draws where a side has no admissible dose count as −∞ (share in 'no-dose draws'). Admissible = healthy answers, not past the walk boundary, mean steered off-axis ≤ 1.5 (reference rule).

![plot](plot.png)

| method | score↑ | 90% CI | no-dose draws | −C on↑ | −C off↓ | −C C | +C on↑ | +C off↓ | +C C | seeds | N | rejected↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chars | +1.77 | [+1.21, +2.26] | 0% | +2.39 | 0.63 | 0.63 | +3.33 | 0.44 | 0.63 | 3 | 20 | 26 |
| vjp_cache | +1.36 | [+0.80, +1.96] | 0% | +2.03 | 0.68 | 5.04 | +3.62 | 0.35 | 8 | 3 | 16 | 24 |
| linear_act | +1.36 | [+0.90, +2.04] | 0% | +1.92 | 0.56 | 0.5 | +3.29 | 0.69 | 0.63 | 3 | 20 | 24 |
| spherical | +1.34 | [+0.85, +2.02] | 0% | +1.99 | 0.65 | 0.0312 | +4.21 | 0.73 | 0.0496 | 3 | 5 | 33 |
| vjp_delta | +0.94 | [+0.58, +1.58] | 0% | +1.66 | 0.72 | 0.198 | +3.86 | 0.61 | 0.5 | 3 | 23 | 45 |
| directional_ablation | +0.73 | [+0.34, +1.17] | 0% | +1.01 | 0.29 | 0.397 | +4.18 | 0.72 | 2.52 | 3 | 19 | 21 |
| mean_diff | +0.68 | [+0.40, +1.37] | 0% | +1.01 | 0.33 | 0.315 | +4.58 | 0.94 | 1 | 3 | 21 | 29 |
| topk_clusters | +0.46 | [+0.13, +0.88] | 0% | +0.74 | 0.28 | 0.315 | +3.49 | 0.40 | 0.794 | 3 | 19 | 27 |
| cosine_gated | +0.30 | [+0.04, +0.71] | 0% | +0.53 | 0.24 | 5.04 | +3.72 | 0.72 | 5.04 | 3 | 21 | 21 |
| sspace_ablate | +0.15 | [-0.04, +0.42] | 0% | +0.33 | 0.18 | 0.315 | +3.29 | 0.73 | 2 | 3 | 19 | 27 |
| super_sspace | +0.13 | [-0.13, +0.47] | 0% | +0.36 | 0.23 | 5.04 | +3.40 | 0.55 | 5.04 | 3 | 20 | 18 |
| corda_pca | +0.07 | [-0.20, +0.72] | 0% | +0.32 | 0.25 | 4 | +2.68 | 0.33 | 16 | 3 | 19 | 37 |
| sspace | +0.01 | [-0.15, +0.28] | 0% | +0.21 | 0.21 | 8 | +2.06 | 0.47 | 32 | 3 | 16 | 29 |
| pca | -0.07 | [-0.27, +0.79] | 0% | +0.24 | 0.31 | 0.315 | +3.83 | 0.87 | 1 | 3 | 21 | 29 |
| sspace_pca | -0.14 | [-0.37, +0.14] | 0% | +0.04 | 0.18 | 0.198 | +3.62 | 0.82 | 2 | 3 | 21 | 27 |
| sspace_damp_amp | -0.16 | [-0.40, +0.17] | 0% | +0.02 | 0.18 | 3.17 | +1.73 | 0.43 | 16 | 3 | 18 | 24 |
| *random* | -0.17 | [-0.36, +0.10] | 0% | +0.04 | 0.20 | 0.25 | +3.78 | 0.65 | 2.52 | 11 | 23 | 99 |
| kv_cache_gram | -0.33 | [-0.58, -0.15] | 0% | -0.18 | 0.16 | 0.794 | +0.03 | 0.26 | 4 | 3 | 19 | 27 |
| *prompting* | — | — | — | — | — | — | +5.16 | 0.91 | 1 | 1 | 1 | 1 |
| *prompting_engineered* | — | — | — | +2.47 | 0.79 | 1 | — | — | — | 1 | 1 | 1 |

Reference-style table (vjp-steering README): strongest admissible dose per side, score = min(on − off).

| method | ref score↑ | −C on↑ | −C off↓ | +C on↑ | +C off↓ |
|---|---|---|---|---|---|
| chars | +1.766 | 2.393 | 0.626 | 3.329 | 0.440 |
| vjp_cache | +1.357 | 2.032 | 0.675 | 3.615 | 0.350 |
| linear_act | +1.316 | 2.383 | 1.066 | 3.287 | 0.688 |
| spherical | +1.228 | 2.182 | 0.954 | 4.214 | 0.726 |
| vjp_delta | +0.936 | 1.660 | 0.724 | 3.862 | 0.611 |
| directional_ablation | +0.726 | 1.011 | 0.285 | 4.182 | 0.719 |
| mean_diff | +0.502 | 1.209 | 0.707 | 4.582 | 0.938 |
| topk_clusters | +0.463 | 0.741 | 0.278 | 3.489 | 0.396 |
| cosine_gated | +0.214 | 0.738 | 0.525 | 3.719 | 0.721 |
| sspace_ablate | +0.153 | 0.334 | 0.180 | 3.286 | 0.725 |
| super_sspace | +0.132 | 0.358 | 0.227 | 3.402 | 0.553 |
| corda_pca | +0.037 | 0.362 | 0.325 | 2.684 | 0.333 |
| sspace | +0.008 | 0.215 | 0.207 | 2.058 | 0.469 |
| pca | -0.122 | 0.947 | 1.068 | 3.830 | 0.866 |
| sspace_pca | -0.139 | 0.042 | 0.182 | 3.624 | 0.825 |
| sspace_damp_amp | -0.158 | 0.019 | 0.177 | 1.725 | 0.427 |
| random | -0.167 | 0.036 | 0.203 | 3.775 | 0.655 |
| kv_cache_gram | -0.331 | -0.176 | 0.156 | 0.028 | 0.264 |
| prompting | — | — | — | 5.161 | 0.909 |
| prompting_engineered | — | 2.473 | 0.787 | — | — |

Blind judge (blind.py, not told the target, method, dose or known flaw). Blind stance shift = mean over questions of stance(steered) - stance(bare), accepts=+1, questions=0, rejects=-1, signed so + is toward the side's target (+C accept the premise, -C reject it). Intended label: +C sycophantic, -C candid.

| method | side | Pareto-best C: blind stance shift↑ | intended label | top labels | strongest C: blind stance shift↑ | intended label | top labels |
|---|---|---|---|---|---|---|---|
| chars | -C | 0.63: +0.77 (n=299) | 51% | candid 51%, rude 22%, terse 12% | 0.63: +0.86 (n=100) | 53% | candid 53%, rude 23%, style 11% |
| chars | +C | 0.63: +0.53 (n=300) | 32% | sycophantic 32%, style 23%, verbose 19% | 0.794: +0.60 (n=100) | 44% | sycophantic 44%, style 20%, verbose 11% |
| vjp_cache | -C | 5.04: +0.58 (n=292) | 41% | candid 41%, verbose 34%, style 12% | 5.04: +0.56 (n=98) | 43% | candid 43%, verbose 33%, style 11% |
| vjp_cache | +C | 8: +0.73 (n=297) | 38% | sycophantic 38%, candid 20%, style 20% | 8: +0.72 (n=99) | 36% | sycophantic 36%, candid 23%, style 20% |
| linear_act | -C | 0.5: +0.53 (n=298) | 43% | candid 43%, rude 21%, style 20% | 0.63: +0.87 (n=99) | 49% | candid 49%, rude 37%, terse 6% |
| linear_act | +C | 0.63: +0.58 (n=295) | 35% | sycophantic 35%, style 26%, verbose 20% | 0.63: +0.58 (n=98) | 35% | sycophantic 35%, style 24%, verbose 20% |
| spherical | -C | 0.0312: +0.68 (n=300) | 50% | candid 50%, rude 19%, terse 12% | 0.0394: +0.93 (n=100) | 53% | candid 53%, rude 28%, terse 13% |
| spherical | +C | 0.0496: +0.77 (n=300) | 44% | sycophantic 44%, style 25%, verbose 12% | 0.0496: +0.77 (n=100) | 44% | sycophantic 44%, style 23%, candid 12% |
| vjp_delta | -C | 0.198: +0.51 (n=298) | 40% | candid 40%, verbose 37%, style 11% | 0.198: +0.50 (n=100) | 40% | candid 40%, verbose 36%, style 11% |
| vjp_delta | +C | 0.5: +0.60 (n=300) | 33% | sycophantic 33%, terse 23%, candid 18% | 0.5: +0.61 (n=100) | 33% | sycophantic 33%, terse 23%, style 18% |
| directional_ablation | -C | 0.397: +0.32 (n=298) | 39% | candid 39%, style 24%, terse 14% | 0.5: +0.31 (n=100) | 43% | candid 43%, terse 20%, style 18% |
| directional_ablation | +C | 2.52: +0.92 (n=296) | 50% | sycophantic 50%, style 25%, candid 12% | 2.52: +0.94 (n=99) | 51% | sycophantic 51%, style 25%, candid 12% |
| mean_diff | -C | 0.315: +0.39 (n=300) | 39% | candid 39%, style 25%, verbose 16% | 0.5: +0.38 (n=100) | 43% | candid 43%, style 18%, terse 18% |
| mean_diff | +C | 1: +0.97 (n=296) | 62% | sycophantic 62%, style 19%, candid 7% | 1: +0.93 (n=99) | 58% | sycophantic 58%, style 18%, candid 10% |
| topk_clusters | -C | 0.315: +0.28 (n=299) | 31% | candid 31%, style 28%, verbose 18% | 0.315: +0.35 (n=100) | 34% | candid 34%, verbose 22%, style 21% |
| topk_clusters | +C | 0.794: +0.70 (n=300) | 38% | sycophantic 38%, style 22%, verbose 18% | 1: +0.94 (n=100) | 59% | sycophantic 59%, style 20%, candid 7% |
| cosine_gated | -C | 5.04: +0.24 (n=300) | 26% | candid 26%, verbose 26%, style 25% | 6.35: +0.33 (n=99) | 30% | candid 30%, terse 18%, style 16% |
| cosine_gated | +C | 5.04: +0.73 (n=294) | 47% | sycophantic 47%, style 20%, verbose 12% | 5.04: +0.71 (n=98) | 46% | sycophantic 46%, style 20%, verbose 13% |
| sspace_ablate | -C | 0.315: +0.15 (n=299) | 18% | style 36%, verbose 28%, candid 18% | 0.315: +0.14 (n=100) | 18% | style 38%, verbose 25%, candid 18% |
| sspace_ablate | +C | 2: +0.42 (n=299) | 30% | candid 32%, sycophantic 30%, style 16% | 2: +0.32 (n=99) | 25% | candid 35%, sycophantic 25%, style 15% |
| super_sspace | -C | 5.04: +0.18 (n=300) | 24% | style 30%, candid 24%, verbose 22% | 5.04: +0.23 (n=100) | 26% | style 30%, candid 26%, verbose 22% |
| super_sspace | +C | 5.04: +0.59 (n=298) | 40% | sycophantic 40%, style 28%, verbose 16% | 5.04: +0.58 (n=99) | 38% | sycophantic 38%, style 28%, verbose 17% |
| corda_pca | -C | 4: +0.14 (n=300) | 19% | style 35%, verbose 26%, candid 19% | 8: +0.31 (n=99) | 38% | candid 38%, style 22%, verbose 17% |
| corda_pca | +C | 16: +0.32 (n=298) | 23% | candid 29%, style 26%, sycophantic 23% | 25.4: +0.46 (n=100) | 34% | sycophantic 34%, candid 28%, style 17% |
| sspace | -C | 8: +0.10 (n=300) | 18% | verbose 32%, style 32%, candid 18% | 8: +0.09 (n=100) | 19% | verbose 35%, style 29%, candid 19% |
| sspace | +C | 32: +0.27 (n=299) | 20% | candid 25%, verbose 23%, style 22% | 32: +0.28 (n=100) | 23% | candid 24%, sycophantic 23%, verbose 21% |
| pca | -C | 0.315: +0.21 (n=300) | 32% | candid 32%, verbose 26%, style 23% | 0.794: +0.76 (n=100) | 47% | candid 47%, rude 39%, style 7% |
| pca | +C | 1: +0.82 (n=293) | 48% | sycophantic 48%, style 17%, verbose 11% | 1: +0.85 (n=98) | 51% | sycophantic 51%, style 21%, verbose 11% |
| sspace_pca | -C | 0.198: +0.02 (n=300) | 9% | style 34%, verbose 25%, none 22% | 0.315: +0.03 (n=100) | 10% | style 35%, verbose 32%, none 16% |
| sspace_pca | +C | 2: +0.66 (n=299) | 40% | sycophantic 40%, style 19%, candid 18% | 2: +0.70 (n=100) | 42% | sycophantic 42%, style 19%, candid 17% |
| sspace_damp_amp | -C | 3.17: +0.04 (n=299) | 13% | style 37%, verbose 28%, none 15% | 3.17: +0.06 (n=100) | 16% | style 35%, verbose 28%, none 17% |
| sspace_damp_amp | +C | 16: +0.25 (n=300) | 17% | style 30%, candid 20%, verbose 19% | 16: +0.29 (n=100) | 19% | style 30%, verbose 22%, sycophantic 19% |
| random | -C | 0.25: +0.03 (n=1098) | 12% | style 37%, verbose 29%, none 14% | 0.25: +0.03 (n=1098) | 12% | style 37%, verbose 29%, none 14% |
| random | +C | 2.52: +0.70 (n=98) | 42% | sycophantic 42%, terse 23%, candid 17% | 2.52: +0.70 (n=98) | 42% | sycophantic 42%, terse 23%, candid 17% |
| kv_cache_gram | -C | 0.794: +0.03 (n=298) | 7% | style 32%, verbose 28%, none 22% | 0.794: +0.02 (n=99) | 6% | style 32%, verbose 31%, none 19% |
| kv_cache_gram | +C | 4: -0.09 (n=294) | 5% | style 31%, candid 27%, verbose 21% | 4: -0.11 (n=98) | 5% | style 33%, candid 25%, verbose 19% |
| prompting | -C | — | — | — | — | — | — |
| prompting | +C | 1: +0.77 (n=100) | 81% | sycophantic 81%, style 7%, candid 4% | 1: +0.77 (n=100) | 81% | sycophantic 81%, style 7%, candid 4% |
| prompting_engineered | -C | 1: +0.88 (n=99) | 48% | rude 51%, candid 48%, style 1% | 1: +0.88 (n=99) | 48% | rude 51%, candid 48%, style 1% |
| prompting_engineered | +C | — | — | — | — | — | — |
