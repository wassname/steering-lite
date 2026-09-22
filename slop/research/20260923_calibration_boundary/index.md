# Signed dose boundary from saved BS-bench outputs

PI/gpt-6-sol · Offline reconstruction from the completed 20-question evaluation and four disjoint transfer cases. No new generation or judgments.

## Best generation-healthy measured final score

One row per method/sign and per random seed/sign. Score = directed intended − 4 × mean absolute off-axis change. Selection is within each measured three-point dose sweep; it is not a comparison adjusted for winner selection. Bare has algebraic score 0; prompting has one +C point and no magnitude. A 4-question calibration score never enters this table.

### +C

| evidence | score↑ | intended↑ | abs off↓ | coefficient | best × | eligible/total |
|:--|--:|--:|--:|--:|--:|--:|
| [random seed0](../../../outputs/bsbench-v2/results/evidence/7c4c45a53af68697.html) | +1.35 | +2.63 | 0.32 | +3.10 | 1 | 3/3 |
| [vjp_delta](../../../outputs/bsbench-v2/results/evidence/b4a748b7754f1d33.html) | +1.33 | +2.64 | 0.33 | +1.03 | 1.2 | 3/3 |
| [random seed1](../../../outputs/bsbench-v2/results/evidence/3b0a5ce2d0e31773.html) | +0.80 | +2.30 | 0.38 | +3.33 | 1 | 3/3 |
| [prompting](../../../outputs/bsbench-v2/results/evidence/0dc1677eb7728818.html) | +0.78 | +4.66 | 0.97 | — | — | 1/1 |
| [vjp_cache](../../../outputs/bsbench-v2/results/evidence/988c571f9697b933.html) | +0.73 | +1.94 | 0.30 | +3.82 | 1.2 | 3/3 |
| [random seed2](../../../outputs/bsbench-v2/results/evidence/6c1871fd3a776890.html) | +0.58 | +2.08 | 0.38 | +3.18 | 1 | 3/3 |
| [mean_diff](../../../outputs/bsbench-v2/results/evidence/01c3dad8fdefa633.html) | -0.28 | +1.83 | 0.53 | +1.74 | 1 | 2/3 |
| [pca](../../../outputs/bsbench-v2/results/evidence/dc3d82f612e70d9b.html) | -0.30 | +0.63 | 0.23 | +1.87 | 1.2 | 3/3 |
| [random seed3](../../../outputs/bsbench-v2/results/evidence/b859827b4823eed1.html) | -0.36 | +2.08 | 0.61 | +4.24 | 1.2 | 2/3 |
| [random seed4](../../../outputs/bsbench-v2/results/evidence/41fee9ecc4d7ee57.html) | -0.57 | +0.92 | 0.37 | +2.30 | 0.8 | 3/3 |
| [kv_cache_gram](../../../outputs/bsbench-v2/results/evidence/f0a78432d6e1e840.html) | -1.06 | +0.15 | 0.30 | +0.70 | 0.8 | 3/3 |

### -C

| evidence | score↑ | intended↑ | abs off↓ | coefficient | best × | eligible/total |
|:--|--:|--:|--:|--:|--:|--:|
| [vjp_delta](../../../outputs/bsbench-v2/results/evidence/4b45fefaf81c03e8.html) | -0.23 | +2.20 | 0.61 | -0.63 | 0.8 | 2/3 |
| [vjp_cache](../../../outputs/bsbench-v2/results/evidence/5489351493017157.html) | -0.25 | +1.57 | 0.46 | -2.83 | 1 | 2/3 |
| [random seed4](../../../outputs/bsbench-v2/results/evidence/f343911968edd1bc.html) | -1.39 | +0.07 | 0.37 | -2.68 | 0.8 | 1/3 |
| [mean_diff](../../../outputs/bsbench-v2/results/evidence/552c466a5b2db217.html) | -1.60 | -0.31 | 0.32 | -1.22 | 0.8 | 3/3 |
| [pca](../../../outputs/bsbench-v2/results/evidence/bf064ccf0ed6ba76.html) | -1.68 | -0.52 | 0.29 | -1.12 | 0.8 | 3/3 |
| [kv_cache_gram](../../../outputs/bsbench-v2/results/evidence/610ef725c7712b03.html) | -1.70 | -0.72 | 0.24 | -0.64 | 0.8 | 2/3 |
| [random seed1](../../../outputs/bsbench-v2/results/evidence/1b8c66656894d8ec.html) | -2.20 | -0.04 | 0.54 | -2.52 | 0.8 | 3/3 |
| [random seed2](../../../outputs/bsbench-v2/results/evidence/597c7cd4b93e4137.html) | -2.77 | -0.51 | 0.57 | -2.75 | 0.8 | 3/3 |
| [random seed0](../../../outputs/bsbench-v2/results/evidence/6aca5345b4635b6b.html) | -2.84 | -0.66 | 0.54 | -3.38 | 1 | 1/3 |
| [random seed3](../../../outputs/bsbench-v2/results/evidence/9bc526ea3c116a4a.html) | -4.78 | -2.61 | 0.54 | -2.43 | 0.8 | 3/3 |

## Predicted 1× versus the three measured evaluation doses

Health order is 0.8×/1×/1.2×; H = no answer-level reason, F = at least one reason. Highest healthy and first higher failed are observed coefficients, not extrapolated thresholds. Regret is best healthy score minus 1× score. It is absent if 1× failed. KL error is the signed 1× solve’s absolute RMS-KL error in nats.

| evidence | health | 1× C | highest H C | first higher F C | boundary | KL err↓ | regret↓ |
|:--|:--:|--:|--:|--:|:--|--:|--:|
| [kv_cache_gram +C](../../../outputs/bsbench-v2/results/evidence/e90cbd066228f254.html) | HHH | +0.88 | +1.05 | — | all-healthy | 0.001 | +0.65 |
| [mean_diff +C](../../../outputs/bsbench-v2/results/evidence/01c3dad8fdefa633.html) | HHF | +1.74 | +1.74 | +2.09 | bracketed | 0.023 | +0.00 |
| [pca +C](../../../outputs/bsbench-v2/results/evidence/d51f251ac2ae02a7.html) | HHH | +1.56 | +1.87 | — | all-healthy | 0.004 | +0.19 |
| [random seed0 +C](../../../outputs/bsbench-v2/results/evidence/7c4c45a53af68697.html) | HHH | +3.10 | +3.73 | — | all-healthy | 0.042 | +0.00 |
| [random seed1 +C](../../../outputs/bsbench-v2/results/evidence/3b0a5ce2d0e31773.html) | HHH | +3.33 | +4.00 | — | all-healthy | 0.019 | +0.00 |
| [random seed2 +C](../../../outputs/bsbench-v2/results/evidence/6c1871fd3a776890.html) | HHH | +3.18 | +3.82 | — | all-healthy | 0.046 | +0.00 |
| [random seed3 +C](../../../outputs/bsbench-v2/results/evidence/afc7f89f08c1f1c8.html) | FHH | +3.53 | +4.24 | — | nonmonotonic | 0.031 | +1.05 |
| [random seed4 +C](../../../outputs/bsbench-v2/results/evidence/0f9e8b11a4552016.html) | HHH | +2.87 | +3.44 | — | all-healthy | 0.022 | +0.27 |
| [vjp_cache +C](../../../outputs/bsbench-v2/results/evidence/5ed23c5e994b9e0b.html) | HHH | +3.19 | +3.82 | — | all-healthy | 0.012 | +0.02 |
| [vjp_delta +C](../../../outputs/bsbench-v2/results/evidence/6e8cde4952c3bd96.html) | HHH | +0.86 | +1.03 | — | all-healthy | 0.038 | +0.26 |
| [kv_cache_gram -C](../../../outputs/bsbench-v2/results/evidence/29f5265257ae0529.html) | HHF | -0.81 | -0.81 | -0.97 | bracketed | 0.006 | +1.19 |
| [mean_diff -C](../../../outputs/bsbench-v2/results/evidence/ae57a919102abd12.html) | HHH | -1.53 | -1.83 | — | all-healthy | 0.044 | +0.62 |
| [pca -C](../../../outputs/bsbench-v2/results/evidence/52041a579e20cc90.html) | HHH | -1.40 | -1.68 | — | all-healthy | 0.007 | +0.56 |
| [random seed0 -C](../../../outputs/bsbench-v2/results/evidence/6aca5345b4635b6b.html) | FHF | -3.38 | -3.38 | -4.06 | nonmonotonic | 0.030 | +0.00 |
| [random seed1 -C](../../../outputs/bsbench-v2/results/evidence/8a8fe41a116a556e.html) | HHH | -3.15 | -3.78 | — | all-healthy | 0.013 | +0.70 |
| [random seed2 -C](../../../outputs/bsbench-v2/results/evidence/107ca09708708538.html) | HHH | -3.44 | -4.12 | — | all-healthy | 0.006 | +0.29 |
| [random seed3 -C](../../../outputs/bsbench-v2/results/evidence/761e23b76f030179.html) | HHH | -3.04 | -3.64 | — | all-healthy | 0.001 | +0.68 |
| [random seed4 -C](../../../outputs/bsbench-v2/results/evidence/f76fbfba23c1795b.html) | HFF | -3.35 | -2.68 | -3.35 | bracketed | 0.014 | — |
| [vjp_cache -C](../../../outputs/bsbench-v2/results/evidence/5489351493017157.html) | HHF | -2.83 | -2.83 | -3.39 | bracketed | 0.002 | +0.00 |
| [vjp_delta -C](../../../outputs/bsbench-v2/results/evidence/35f2648766897402.html) | HHF | -0.79 | -0.79 | -0.95 | bracketed | 0.035 | +0.94 |

## Four disjoint transfer cases (two questions each)

Transfer has generation health and RMS-KL but no behavioral judgment or score. Full 80 signed case rows, including predicted coefficient, three health flags, first failed measurement and full precision KL, are in [boundary-20q-and-transfer.csv](boundary-20q-and-transfer.csv).

| case | groups | all H | bracketed | KL within .05 | 1× failed |
|:--|--:|--:|--:|--:|--:|
| bsbench-v2-heldout-a | 20 | 18 | 2 | 18 | 0 |
| bsbench-v2-heldout-b | 20 | 19 | 1 | 19 | 0 |
| paper-native-false-claim-agreement-a | 20 | 17 | 3 | 16 | 0 |
| paper-native-false-claim-agreement-b | 20 | 17 | 3 | 16 | 0 |

## Separate four-question candidate grid

Candidate scores and healthy magnitudes are measured on only four calibration questions. The 1× prediction comes from the highest candidate magnitude clean in both signs, then one pooled signed RMS-KL target; these per-sign score optima are not final rankings. All 20 per-sign rows are in [calibration-candidates-4q.csv](calibration-candidates-4q.csv).

| method / seed | +C H/grid | −C H/grid | highest both-sign H | target RMS-KL |
|:--|--:|--:|--:|--:|
| kv_cache_gram / 0 | 5/5 | 4/5 | 0.8 | 0.209 |
| mean_diff / 0 | 5/6 | 5/6 | 1.6 | 0.791 |
| pca / 0 | 6/6 | 5/6 | 1.6 | 0.193 |
| random / 0 | 6/7 | 6/7 | 3.2 | 0.979 |
| random / 1 | 7/7 | 6/7 | 3.2 | 0.875 |
| random / 2 | 6/7 | 6/7 | 3.2 | 0.990 |
| random / 3 | 7/7 | 6/7 | 3.2 | 0.993 |
| random / 4 | 6/7 | 6/7 | 3.2 | 0.782 |
| vjp_cache / 0 | 7/7 | 6/7 | 3.2 | 0.213 |
| vjp_delta / 0 | 4/5 | 4/5 | 0.8 | 0.798 |

## Eligibility and interpretation

The [report producer](../../../scripts/run_bsbench_results.py) marks a point healthy when `flags == 0`, with `flags = sum(bool(e["health"]["reasons"]) for e in examples)` (lines 95–101). The [production final producer](../../../scripts/run_bsbench_modal.py) calls `health(tokenizer, [answer])` for each of 168 individual answers (lines 276–283). The [health implementation](../../../src/steering_lite/benchmark/generation.py) requires a sentence-ending `[.!?")]$`, role-leak fraction <0.25, repeated fraction <0.25 and unfinished fraction <0.5 (lines 130–151). One answer means one punctuation-only unfinished flag excludes its whole 20-question dose.

The [pinned reference admissibility producer](../../../docs/vendor/vjp-steering/scripts/export.py) uses `not health["breakdown_reasons"] and not health["post_boundary"] and steered_off_axis <= 1.5` (lines 194–198). Its [health producer](../../../docs/vendor/vjp-steering/scripts/walk.py) evaluates all answers as a cohort, calls the same regex and fraction thresholds (lines 408–438), and sets `post_boundary` only after two consecutive failed magnitudes (lines 247–260). Local final health uses per-answer reasons, so it is stricter than the reference’s cohort threshold; it also has no post-boundary or off-axis criterion. The plan intentionally removes the off-axis cutoff, so do not silently re-add it. Whether to keep per-answer exclusion, use a cohort rate, or treat a complete punctuation-only answer differently is a scientific decision, not a plotting fix.

Eight of the nine locally flagged 20-question doses would pass the reference’s *cohort-fraction health thresholds alone* (see [eight source-linked rows](eligibility-disagreements.csv)). The remaining random seed4 −C at 1.2× has 9/20 role leaks and fails both definitions. This comparison deliberately excludes the reference’s separate off-axis and post-boundary conditions; it does not reclassify the current report. The flagged raw answers include incomplete text, an empty answer, role leaks and a repeated-zero answer. `No\nNo` in transfer is punctuation-only flagged, not by itself proof of truncation.

Two measured evaluation health patterns recover after a lower failed dose: random seed0 −C is FHF, random seed3 +C is FHH. Neither gives a monotone bracket. A row marked all-healthy has no observed upper boundary; an all-failed row would have no measured lower healthy bound. A bracketed row bounds only the sampled doses and predicate, not an exact maximum. The full [final and transfer table](boundary-20q-and-transfer.csv) records every group.

At the current predicate, evaluation: 13/20 all-healthy, 5/20 bracketed, 2/20 nonmonotonic; 1/20 failed at 1×. Transfer: 71/80 all-healthy, 9/80 bracketed, 0/80 failed at 1×. RMS-KL absolute error ≤.05: evaluation 20/20, transfer 69/80. Best measured healthy score is at 0.8× for 10/20, 1× for 6/20 and 1.2× for 4/20. Thus the 1× KL target often matches KL yet does not select the best measured score. These are within-sample maximizations on just three doses, not out-of-sample efficacy.

Most of the 84 all-healthy case/sign groups never measure their upper health boundary; calibrating RMS-KL alone cannot prove it predicts maximum coherent dose. Nine transfer groups and five evaluation groups have a measured first higher failing point. The 11 transfer KL misses include 10 all-healthy groups and one bracketed group, so KL miss and health failure are not interchangeable.

## Cohort-health-only diagnostic

[Side-by-side selection, boundary and newly eligible raw failures](cohort-health-only.md). This is a counterfactual from cached singleton metrics, not a change to the canonical eligibility rule.

## Evidence files

- [All 21 final best rows](best-final.csv), [20 four-question candidate summaries](calibration-candidates-4q.csv), [100 signed evaluation/transfer boundaries](boundary-20q-and-transfer.csv), [eight eligibility disagreements](eligibility-disagreements.csv).
- [Raw measured points](../../../outputs/bsbench-v2/results/measured-points.json), [20-question evidence index](../../../outputs/bsbench-v2/results/index.md), [frozen summary](../../../outputs/bsbench-v2/run-summary.json).
- Calculations: [offline-only script](../../verification/20260923_calibration_boundary_offline.py), [run output](../../verification/20260923_calibration_boundary_summary.log).

— PI/gpt-6-sol
