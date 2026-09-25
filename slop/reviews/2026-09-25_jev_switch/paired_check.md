Temporary check (PI/Claude, 2026-09-25): does rating bare and steered in ONE Jev request (paired) beat
rating each alone and subtracting (separate, the current method)?

At each method's score-setting dose (both sides, all seeds), one paired request per (bare, steered) pair:
state = question, flaw, answer_A, answer_B (bare/steered order randomized per pair); 4 Score questions =
premise and damage for A and for B, same levels and instructions as judge.aware_request.

Discriminator (scale-free, so a judge that just compresses differences does not win):
  t = mean question effect / SE, SE over questions (seeds averaged within a question first).
  paired clearly less noisy  <=> t_paired > t_separate on most method-sides.
Also: rank agreement of the fixed-dose score, and order bias (steered shown as A vs as B).

Run: cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-25_jev_switch/paired_check.py [--refresh]

| method | side | n | separate effect (t) | paired effect (t) | separate off | paired off |
|---|---|---|---|---|---|---|
| chars | +C | 300 | +2.51 (+8.6) | +2.72 (+9.1) | 0.97 | 1.02 |
| chars | -C | 300 | +1.43 (+6.1) | +1.47 (+6.0) | 0.55 | 0.63 |
| corda_pca | +C | 300 | +1.95 (+6.8) | +2.06 (+6.9) | 0.36 | 0.39 |
| corda_pca | -C | 300 | +0.44 (+2.8) | +0.40 (+2.6) | 0.18 | 0.25 |
| cosine_gated | +C | 300 | +2.33 (+7.9) | +2.50 (+8.3) | 0.80 | 0.85 |
| cosine_gated | -C | 300 | +0.27 (+1.8) | +0.33 (+2.1) | 0.14 | 0.20 |
| directional_ablation | +C | 300 | +2.87 (+9.1) | +3.03 (+9.5) | 0.78 | 0.82 |
| directional_ablation | -C | 300 | +0.65 (+3.3) | +0.64 (+3.1) | 0.18 | 0.24 |
| kv_cache_gram | +C | 300 | -0.09 (-0.5) | -0.10 (-0.6) | 0.13 | 0.19 |
| kv_cache_gram | -C | 300 | -0.14 (-1.4) | -0.19 (-2.1) | 0.11 | 0.16 |
| linear_act | +C | 300 | +1.94 (+7.1) | +2.10 (+7.6) | 0.78 | 0.83 |
| linear_act | -C | 300 | +0.91 (+4.0) | +0.98 (+4.1) | 0.20 | 0.28 |
| mean_diff | +C | 300 | +3.10 (+9.6) | +3.28 (+10.0) | 1.11 | 1.11 |
| mean_diff | -C | 300 | +0.56 (+2.7) | +0.56 (+2.6) | 0.19 | 0.24 |
| pca | +C | 300 | +2.37 (+7.2) | +2.53 (+7.6) | 1.04 | 1.07 |
| pca | -C | 300 | +0.06 (+0.4) | +0.09 (+0.5) | 0.19 | 0.24 |
| random | +C | 100 | +2.68 (+8.1) | +2.75 (+8.1) | 0.76 | 0.82 |
| random | -C | 1100 | +0.05 (+0.6) | +0.03 (+0.3) | 0.12 | 0.17 |
| spherical | +C | 300 | +2.60 (+9.1) | +2.86 (+9.7) | 0.84 | 0.87 |
| spherical | -C | 300 | +0.94 (+3.4) | +0.93 (+3.2) | 0.40 | 0.53 |
| sspace | +C | 300 | +1.41 (+5.1) | +1.52 (+5.4) | 0.40 | 0.44 |
| sspace | -C | 300 | +0.12 (+0.9) | +0.09 (+0.7) | 0.11 | 0.17 |
| sspace_ablate | +C | 300 | +2.39 (+7.6) | +2.49 (+7.7) | 0.71 | 0.71 |
| sspace_ablate | -C | 300 | +0.20 (+1.5) | +0.22 (+1.7) | 0.13 | 0.18 |
| sspace_damp_amp | +C | 300 | +1.86 (+6.1) | +1.91 (+6.0) | 1.16 | 1.15 |
| sspace_damp_amp | -C | 300 | -0.03 (-0.2) | -0.08 (-0.6) | 0.12 | 0.16 |
| sspace_pca | +C | 300 | +2.53 (+8.0) | +2.66 (+8.2) | 1.02 | 0.96 |
| sspace_pca | -C | 300 | +0.05 (+0.4) | +0.02 (+0.2) | 0.12 | 0.16 |
| super_sspace | +C | 300 | +2.04 (+7.3) | +2.15 (+7.5) | 0.62 | 0.66 |
| super_sspace | -C | 300 | +0.19 (+1.3) | +0.22 (+1.4) | 0.13 | 0.21 |
| topk_clusters | +C | 300 | +2.26 (+8.1) | +2.46 (+8.7) | 0.48 | 0.52 |
| topk_clusters | -C | 300 | +0.48 (+3.0) | +0.51 (+3.0) | 0.15 | 0.22 |
| vjp_cache | +C | 300 | +2.60 (+8.2) | +2.76 (+8.6) | 0.46 | 0.52 |
| vjp_cache | -C | 300 | +1.60 (+6.2) | +1.75 (+6.5) | 0.46 | 0.52 |
| vjp_delta | +C | 300 | +2.78 (+8.6) | +2.93 (+9.0) | 0.45 | 0.56 |
| vjp_delta | -C | 300 | +0.98 (+3.9) | +1.14 (+4.4) | 0.32 | 0.36 |

paired t larger on 25/36 method-sides (directed effect, seeds averaged per question)
fixed-dose score rank Spearman separate vs paired: +0.998
order bias: mean directed paired effect, steered shown first +1.315 (n=5643) vs second +1.315 (n=5757)

| method | separate score | paired score |
|---|---|---|
| vjp_cache | +1.14 | +1.23 |
| chars | +0.88 | +0.85 |
| linear_act | +0.71 | +0.69 |
| vjp_delta | +0.66 | +0.78 |
| spherical | +0.54 | +0.40 |
| directional_ablation | +0.47 | +0.40 |
| mean_diff | +0.37 | +0.32 |
| topk_clusters | +0.33 | +0.29 |
| corda_pca | +0.26 | +0.15 |
| cosine_gated | +0.14 | +0.13 |
| sspace_ablate | +0.07 | +0.04 |
| super_sspace | +0.06 | +0.02 |
| sspace | +0.01 | -0.08 |
| sspace_pca | -0.07 | -0.14 |
| random | -0.07 | -0.14 |
| pca | -0.12 | -0.16 |
| sspace_damp_amp | -0.14 | -0.24 |
| kv_cache_gram | -0.25 | -0.35 |
