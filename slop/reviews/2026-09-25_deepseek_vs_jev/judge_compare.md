# DeepSeek vs Jev (full)

How far do the DeepSeek and Jev judges agree, and is the gap more than each judge's own noise?

1. Judge noise (test-retest) on answers from admissible doses:
   DeepSeek: pass 0 vs pass 1 (each pass = mean of the AB and BA calls); Jev: re-ask N_RETEST answers.
   Reliability of the per-answer score the results use (DeepSeek: 2 passes, Spearman-Brown; Jev: 1 call).
2. Answer-level agreement between judges, raw and corrected for both judges' noise
   (r / sqrt(rel_ds * rel_jev)): near 1 = same measure plus noise, well below 1 = they measure different things.
3. Method ranking under DeepSeek, Jev on DeepSeek's admissible doses, and Jev with its own admissibility
   (health rule + mean steered Jev damage <= JEV_MAX_DAMAGE), with paired bootstrap draws
   (same seeds and questions for every view) -> Spearman of the rankings and P(same order) per adjacent pair.

## 1. Judge noise: each judge against itself (steered answers at DeepSeek-admissible doses)

| judge | what is compared | n answers | Pearson r | reliability of the score used |
|---|---|--:|--:|--:|
| DeepSeek | pass 0 vs pass 1 on-axis change (each pass = mean of AB and BA) | 120100 | 0.94 | 0.97 (2 passes, Spearman-Brown) |
| Jev | premise level, first call vs re-ask | 400 | 1.00 | 1.00 (1 call) |

## 2. Answer-level agreement between the judges (same answers as above)

on-axis = change from the bare answer toward sycophancy (DeepSeek -10..10 pairwise, Jev premise level -6..6). Corrected r = Pearson r / sqrt(rel_DeepSeek × rel_Jev), the agreement expected if both judges measured the same thing with their measured noise.

| side | n answers | on-axis Pearson r | on-axis Spearman | corrected r | off-axis Spearman | same sign |
|---|--:|--:|--:|--:|--:|--:|
| +C | 62700 | 0.92 | 0.73 | 0.93 | 0.49 | 63% |
| -C | 57400 | 0.90 | 0.64 | 0.92 | 0.48 | 59% |
| both | 120100 | 0.91 | 0.69 | 0.93 | 0.48 | 61% |

Same sign by size of the DeepSeek change (|on-axis|, DeepSeek units); Jev = 0 counts as not the same sign:

| DeepSeek abs(change) | share of answers | same sign | Jev exactly 0 |
|---|--:|--:|--:|
| 0-0.5 | 64% | 45% | 36% |
| 0.5-1 | 7% | 71% | 13% |
| 1-2 | 6% | 81% | 9% |
| 2-4 | 6% | 89% | 5% |
| 4-99 | 17% | 99% | 0% |

## 3. Method ranking

Score = min over ±C of (on − off) at the best admissible dose, in each judge's own units. Rank agreement: Spearman over 18 methods, point estimate and 90% interval over 300 paired bootstrap draws (seeds then questions; the same draws for every view). 'Jev, own doses' replaces DeepSeek's admissibility cap (steered off-axis ≤ 1.5 of 5) with steered Jev damage ≤ 1.5 of 4; the health rule and walk boundary are the same.

'DeepSeek units': Jev on-axis × 1.80 and off-axis × 1.47 (ratio of the two judges' standard deviations over the answers in section 2), so the 1:1 score weighs damage the same under both judges. Unscaled, Jev's 1:1 score weighs damage about 1.2× more than DeepSeek's.

- DeepSeek vs Jev, DeepSeek doses: Spearman +0.97 [+0.86, +0.97]
- DeepSeek vs Jev, own doses: Spearman +0.97 [+0.86, +0.97]
- DeepSeek vs Jev, own doses, DeepSeek units: Spearman +0.97 [+0.87, +0.98]

| method | DeepSeek score (rank) | Jev, DeepSeek doses (rank) | Jev, own doses (rank) | Jev, own doses, DeepSeek units (rank) | Jev-own admissible doses / DeepSeek admissible doses |
|---|--:|--:|--:|--:|--:|
| chars | +1.77 (1) | +0.52 (2) | +0.52 (2) | +1.13 (2) | 63 / 62 |
| vjp_cache | +1.36 (2) | +0.61 (1) | +0.61 (1) | +1.26 (1) | 48 / 48 |
| linear_act | +1.36 (3) | +0.45 (3) | +0.45 (3) | +0.88 (3) | 63 / 60 |
| spherical | +1.34 (4) | +0.39 (4) | +0.39 (4) | +0.84 (4) | 18 / 15 |
| vjp_delta | +0.94 (5) | +0.39 (5) | +0.39 (5) | +0.81 (5) | 69 / 69 |
| directional_ablation | +0.73 (6) | +0.27 (6) | +0.27 (6) | +0.55 (6) | 61 / 61 |
| mean_diff | +0.68 (7) | +0.23 (7) | +0.23 (7) | +0.48 (7) | 63 / 63 |
| topk_clusters | +0.46 (8) | +0.18 (8) | +0.18 (8) | +0.38 (8) | 67 / 67 |
| cosine_gated | +0.30 (9) | +0.02 (10) | +0.02 (10) | +0.13 (10) | 63 / 63 |
| sspace_ablate | +0.15 (10) | +0.00 (11) | +0.00 (11) | +0.05 (11) | 60 / 57 |
| super_sspace | +0.13 (11) | -0.04 (13) | -0.04 (13) | -0.03 (13) | 60 / 60 |
| corda_pca | +0.07 (12) | +0.11 (9) | +0.11 (9) | +0.25 (9) | 62 / 61 |
| sspace | +0.01 (13) | -0.02 (12) | -0.02 (12) | +0.00 (12) | 55 / 53 |
| pca | -0.07 (14) | -0.12 (15) | -0.12 (15) | -0.16 (15) | 63 / 63 |
| sspace_pca | -0.14 (15) | -0.12 (16) | -0.12 (16) | -0.17 (16) | 66 / 63 |
| sspace_damp_amp | -0.16 (16) | -0.15 (17) | -0.15 (17) | -0.22 (17) | 59 / 56 |
| random | -0.17 (17) | -0.10 (14) | -0.10 (14) | -0.14 (14) | 235 / 223 |
| kv_cache_gram | -0.33 (18) | -0.24 (18) | -0.24 (18) | -0.40 (18) | 61 / 57 |

Score parts at each side's score-setting dose (on-axis gain toward the side's target, off-axis), DeepSeek | Jev (DeepSeek doses). Jev units are smaller, so compare the on/off ratio, not the raw numbers:

| method | -C DeepSeek on / off | -C Jev on / off | +C DeepSeek on / off | +C Jev on / off |
|---|--:|--:|--:|--:|
| chars | 2.39 / 0.63 (C=0.63) | 1.09 / 0.57 (C=0.63) | 3.33 / 0.44 (C=0.63) | 1.51 / 0.52 (C=0.63) |
| vjp_cache | 2.03 / 0.68 (C=5.04) | 1.09 / 0.47 (C=5.04) | 3.62 / 0.35 (C=8) | 1.87 / 0.45 (C=8) |
| linear_act | 1.92 / 0.56 (C=0.5) | 0.67 / 0.22 (C=0.397) | 3.29 / 0.69 (C=0.63) | 1.43 / 0.68 (C=0.63) |
| spherical | 1.99 / 0.65 (C=0.0312) | 0.80 / 0.40 (C=0.0312) | 4.21 / 0.73 (C=0.0496) | 1.94 / 0.72 (C=0.0496) |
| vjp_delta | 1.66 / 0.72 (C=0.198) | 0.70 / 0.31 (C=0.157) | 3.86 / 0.61 (C=0.5) | 1.98 / 0.52 (C=0.5) |
| directional_ablation | 1.01 / 0.29 (C=0.397) | 0.46 / 0.20 (C=0.5) | 4.18 / 0.72 (C=2.52) | 2.06 / 0.72 (C=2.52) |
| mean_diff | 1.01 / 0.33 (C=0.315) | 0.42 / 0.19 (C=0.397) | 4.58 / 0.94 (C=1) | 2.29 / 0.92 (C=1) |
| topk_clusters | 0.74 / 0.28 (C=0.315) | 0.34 / 0.16 (C=0.315) | 3.49 / 0.40 (C=0.794) | 1.69 / 0.46 (C=0.794) |
| cosine_gated | 0.53 / 0.24 (C=5.04) | 0.17 / 0.15 (C=3.17) | 3.72 / 0.72 (C=5.04) | 1.74 / 0.71 (C=5.04) |
| sspace_ablate | 0.33 / 0.18 (C=0.315) | 0.14 / 0.14 (C=0.315) | 3.29 / 0.73 (C=2) | 1.75 / 0.72 (C=2) |
| super_sspace | 0.36 / 0.23 (C=5.04) | 0.10 / 0.14 (C=5.04) | 3.40 / 0.55 (C=5.04) | 1.52 / 0.57 (C=5.04) |
| corda_pca | 0.32 / 0.25 (C=4) | 0.28 / 0.17 (C=6.35) | 2.68 / 0.33 (C=16) | 1.37 / 0.40 (C=16) |
| sspace | 0.21 / 0.21 (C=8) | 0.10 / 0.12 (C=6.35) | 2.06 / 0.47 (C=32) | 0.99 / 0.43 (C=32) |
| pca | 0.24 / 0.31 (C=0.315) | 0.02 / 0.13 (C=0.0992) | 3.83 / 0.87 (C=1) | 1.69 / 0.90 (C=1) |
| sspace_pca | 0.04 / 0.18 (C=0.198) | 0.00 / 0.12 (C=0.198) | 3.62 / 0.82 (C=2) | 1.83 / 0.89 (C=2) |
| sspace_damp_amp | 0.02 / 0.18 (C=3.17) | -0.02 / 0.13 (C=2.52) | 1.73 / 0.43 (C=16) | 0.80 / 0.48 (C=16) |
| random | 0.04 / 0.20 (C=0.25) | 0.02 / 0.13 (C=0.25) | 3.78 / 0.65 (C=2.52) | 1.83 / 0.79 (C=2.52) |
| kv_cache_gram | -0.18 / 0.16 (C=0.794) | -0.14 / 0.10 (C=0.794) | 0.03 / 0.26 (C=4) | -0.08 / 0.14 (C=1.26) |

P(first method scores higher than second) over the paired draws, for DeepSeek-adjacent pairs:

| pair (DeepSeek order) | DeepSeek | Jev, DeepSeek doses | Jev, own doses | Jev, own doses, DeepSeek units |
|---|--:|--:|--:|--:|
| chars > vjp_cache | 77% | 33% | 33% | 35% |
| vjp_cache > linear_act | 44% | 82% | 80% | 81% |
| linear_act > spherical | 52% | 57% | 57% | 52% |
| spherical > vjp_delta | 80% | 47% | 47% | 48% |
| vjp_delta > directional_ablation | 80% | 74% | 74% | 79% |
| directional_ablation > mean_diff | 46% | 58% | 58% | 58% |
| mean_diff > topk_clusters | 82% | 65% | 65% | 65% |
| topk_clusters > cosine_gated | 68% | 75% | 75% | 73% |
| cosine_gated > sspace_ablate | 75% | 65% | 65% | 68% |
| sspace_ablate > super_sspace | 56% | 60% | 60% | 61% |
| super_sspace > corda_pca | 46% | 26% | 26% | 25% |
| corda_pca > sspace | 65% | 77% | 77% | 78% |
| sspace > pca | 34% | 71% | 71% | 67% |
| pca > sspace_pca | 79% | 62% | 62% | 66% |
| sspace_pca > sspace_damp_amp | 44% | 48% | 48% | 48% |
| sspace_damp_amp > random | 52% | 36% | 36% | 38% |
| random > kv_cache_gram | 89% | 96% | 96% | 96% |

Admissibility of (method, seed, dose, side) points: both 1201, DeepSeek only 0, Jev only 35, neither 532.

