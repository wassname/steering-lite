Temporary check (PI/Claude, 2026-09-26), wassname's design: Jev rates the DIFFERENCE between two answers
directly, asked in both orders and combined to cancel direction bias:
  d = (rate(A=bare, B=steered) - rate(A=steered, B=bare)) / 2,  rate in -4..+4 (9 levels, 4 = same)
for premise (on-axis) and damage (off-axis = |d_damage|). Compared with the current method (rate each answer
alone, subtract), at every method's score-setting dose, all seeds.

Discriminators: t = mean question effect / SE over questions (scale-free); fixed-dose score ranking;
direction bias = mean of rate(AB) + rate(BA) (0 if the judge is antisymmetric).

Run: cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-25_jev_switch/diff_check.py [--refresh]

| method | side | n | alone effect (t) | diff effect (t) | alone off | diff off |
|---|---|---|---|---|---|---|
| chars | +C | 300 | +2.51 (+8.6) | +1.93 (+10.8) | 0.97 | 2.32 |
| chars | -C | 300 | +1.43 (+6.1) | +1.78 (+12.1) | 0.55 | 1.26 |
| corda_pca | +C | 300 | +1.95 (+6.8) | +1.13 (+5.4) | 0.36 | 1.20 |
| corda_pca | -C | 300 | +0.44 (+2.8) | +0.35 (+3.4) | 0.18 | 0.52 |
| cosine_gated | +C | 300 | +2.33 (+7.9) | +1.82 (+9.7) | 0.80 | 1.95 |
| cosine_gated | -C | 300 | +0.27 (+1.8) | +0.36 (+3.1) | 0.14 | 0.48 |
| directional_ablation | +C | 300 | +2.87 (+9.1) | +2.02 (+10.0) | 0.78 | 2.08 |
| directional_ablation | -C | 300 | +0.65 (+3.3) | +0.73 (+4.7) | 0.18 | 0.56 |
| kv_cache_gram | +C | 300 | -0.09 (-0.5) | -0.16 (-1.3) | 0.13 | 0.36 |
| kv_cache_gram | -C | 300 | -0.14 (-1.4) | -0.11 (-1.2) | 0.11 | 0.34 |
| linear_act | +C | 300 | +1.94 (+7.1) | +1.63 (+9.2) | 0.78 | 1.82 |
| linear_act | -C | 300 | +0.91 (+4.0) | +0.99 (+6.4) | 0.20 | 0.72 |
| mean_diff | +C | 300 | +3.10 (+9.6) | +2.20 (+11.3) | 1.11 | 2.53 |
| mean_diff | -C | 300 | +0.56 (+2.7) | +0.65 (+4.1) | 0.19 | 0.63 |
| pca | +C | 300 | +2.37 (+7.2) | +1.66 (+7.3) | 1.04 | 2.26 |
| pca | -C | 300 | +0.06 (+0.4) | +0.33 (+2.3) | 0.19 | 0.58 |
| random | +C | 100 | +2.68 (+8.1) | +1.39 (+5.3) | 0.76 | 2.15 |
| random | -C | 1100 | +0.05 (+0.6) | +0.06 (+0.9) | 0.12 | 0.36 |
| spherical | +C | 300 | +2.60 (+9.1) | +2.09 (+11.7) | 0.84 | 2.10 |
| spherical | -C | 300 | +0.94 (+3.4) | +1.39 (+7.7) | 0.40 | 1.05 |
| sspace | +C | 300 | +1.41 (+5.1) | +1.02 (+5.1) | 0.40 | 1.35 |
| sspace | -C | 300 | +0.12 (+0.9) | +0.11 (+1.2) | 0.11 | 0.35 |
| sspace_ablate | +C | 300 | +2.39 (+7.6) | +1.47 (+6.1) | 0.71 | 2.00 |
| sspace_ablate | -C | 300 | +0.20 (+1.5) | +0.26 (+2.5) | 0.13 | 0.40 |
| sspace_damp_amp | +C | 300 | +1.86 (+6.1) | +1.04 (+4.5) | 1.16 | 2.18 |
| sspace_damp_amp | -C | 300 | -0.03 (-0.2) | -0.06 (-0.6) | 0.12 | 0.32 |
| sspace_pca | +C | 300 | +2.53 (+8.0) | +1.91 (+9.1) | 1.02 | 2.19 |
| sspace_pca | -C | 300 | +0.05 (+0.4) | +0.04 (+0.4) | 0.12 | 0.36 |
| super_sspace | +C | 300 | +2.04 (+7.3) | +1.64 (+9.2) | 0.62 | 1.64 |
| super_sspace | -C | 300 | +0.19 (+1.3) | +0.24 (+2.0) | 0.13 | 0.47 |
| topk_clusters | +C | 300 | +2.26 (+8.1) | +1.85 (+11.0) | 0.48 | 1.49 |
| topk_clusters | -C | 300 | +0.48 (+3.0) | +0.55 (+4.6) | 0.15 | 0.52 |
| vjp_cache | +C | 300 | +2.60 (+8.2) | +1.70 (+7.7) | 0.46 | 1.59 |
| vjp_cache | -C | 300 | +1.60 (+6.2) | +1.16 (+6.3) | 0.46 | 0.98 |
| vjp_delta | +C | 300 | +2.78 (+8.6) | +1.69 (+7.4) | 0.45 | 1.82 |
| vjp_delta | -C | 300 | +0.98 (+3.9) | +0.81 (+4.5) | 0.32 | 0.83 |

|t| ratio diff/alone over method-sides with |t_alone|>2: median 1.190 (min 0.65, max 2.29, n=26)
fixed-dose score rank Spearman alone vs diff: +0.701
rank Spearman alone vs mix (diff premise, alone damage): +0.934
direction bias, mean rate(AB)+rate(BA) (0 = antisymmetric): premise +0.019 (sd 0.48), damage -0.117 (sd 0.45)

| method | alone score | diff score (AB/BA) | mix: diff premise, alone damage |
|---|---|---|---|
| vjp_cache | +1.14 | +0.11 | +0.70 |
| chars | +0.88 | -0.39 | +0.96 |
| linear_act | +0.71 | -0.20 | +0.80 |
| vjp_delta | +0.66 | -0.12 | +0.49 |
| spherical | +0.54 | -0.01 | +0.99 |
| directional_ablation | +0.47 | -0.06 | +0.56 |
| mean_diff | +0.37 | -0.33 | +0.46 |
| topk_clusters | +0.33 | +0.04 | +0.41 |
| corda_pca | +0.26 | -0.17 | +0.18 |
| cosine_gated | +0.14 | -0.13 | +0.22 |
| sspace_ablate | +0.07 | -0.53 | +0.13 |
| super_sspace | +0.06 | -0.23 | +0.11 |
| sspace | +0.01 | -0.33 | -0.00 |
| sspace_pca | -0.07 | -0.32 | -0.08 |
| random | -0.07 | -0.76 | -0.05 |
| pca | -0.12 | -0.60 | +0.15 |
| sspace_damp_amp | -0.14 | -1.14 | -0.18 |
| kv_cache_gram | -0.25 | -0.52 | -0.29 |
