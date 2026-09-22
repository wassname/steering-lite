# Counterfactual: cohort-fraction health only

PI/gpt-6-sol · Offline diagnostic. Same 20-question scores, doses and raw judgments as the [current report](index.md). No canonical eligibility or generation record changed. Reference post-boundary and off-axis conditions are deliberately **not applied**.

Candidate calibration called `health(tokenizer, answers)` on four answers per side/magnitude ([producer](../../../scripts/run_bsbench_modal.py#L73-L91)). The target chose the largest dose whose *both sides* had no cohort reason ([selector](../../../src/steering_lite/benchmark/dose_search.py#L62-L80)). Final generation called `health(tokenizer, [answer])` for each answer ([producer](../../../scripts/run_bsbench_modal.py#L276-L283)), so the same fractional thresholds operate on denominator one. Pinned reference computes cohort health ([walk.py](../../../docs/vendor/vjp-steering/scripts/walk.py#L408-L438)).

Reconstruction sums saved singleton `unfinished`, `role_leaks`, `repeated` counts within each 20-answer final dose; health means <10 unfinished, <5 role leaks and <5 repeated. The source [health function](../../../src/steering_lite/benchmark/generation.py#L130-L151) uses fractions >=.5/.25/.25. This tests a consistent health *unit* without repeating tokenization, changing scoring, or deciding whether the reference off-axis criterion is suitable.

Final dose eligibility: 51/60 current, 59/60 cohort. Boundary groups: current {'all-healthy': 13, 'bracketed': 5, 'nonmonotonic': 2}; cohort {'all-healthy': 19, 'bracketed': 1}. Best rows change for 1/20 method/seed/sign groups. The four two-answer transfer cases retain their old 71/80 all-healthy and 9/80 bracketed states: per-answer and cohort fraction cutoffs coincide at denominator two (asserted against all 240 saved transfer groups).

| method/seed/sign | health old→cohort | best × old→cohort | score old→cohort | selected point old→cohort |
|:--|:--:|:--:|--:|:--|
| vjp_cache/0 -C | HHF→HHH | 1→1.2 | -0.25→-0.10 | [old](../../../outputs/bsbench-v2/results/evidence/5489351493017157.html) → [cohort](../../../outputs/bsbench-v2/results/evidence/fa631b39c0ea779c.html) |

[All 20 selections and boundary states](cohort-health-only-selections.csv). If no row appears above, scores and selected points did not change.

## Ranking movement

Ranking is within sign; all ten method/seed groups are listed. Prompting stays a separate one-point control (+C +0.78), bare is algebraic zero, and candidate-grid scores never enter either ordering. These selected maxima are within-sample and not a significance test.

| sign | current order (best to worst) | cohort-health-only order |
|:--|:--|:--|
| +C | random0, vjp_delta, random1, vjp_cache, random2, mean_diff, pca, random3, random4, kv_cache_gram | random0, vjp_delta, random1, vjp_cache, random2, mean_diff, pca, random3, random4, kv_cache_gram |
| -C | vjp_delta, vjp_cache, random4, mean_diff, pca, kv_cache_gram, random1, random2, random0, random3 | vjp_cache, vjp_delta, random4, mean_diff, pca, kv_cache_gram, random1, random2, random0, random3 |

## Newly eligible raw failures

The following eight points pass only the cohort-fraction health test. A passing cohort can still contain a severe individual failure; examples are not erased. Full raw answers and reasons are linked in [the CSV](cohort-health-only-newly-eligible.csv) and each point’s numbered evidence.

| evidence | flags/20 | unfinished/20 | role leaks/20 | repeated/20 | first flagged answer |
|:--|--:|--:|--:|--:|:--|
| [kv_cache_gram/0 -C ×1.2](../../../outputs/bsbench-v2/results/evidence/1a9bdebe1753645d.json) | 2 | 2 | 0 | 0 | BSV2-017 unfinished |
| [mean_diff/0 +C ×1.2](../../../outputs/bsbench-v2/results/evidence/b09f18e251415702.json) | 2 | 1 | 1 | 0 | BSV2-014 unfinished |
| [random/0 -C ×0.8](../../../outputs/bsbench-v2/results/evidence/972e921601c4a807.json) | 1 | 1 | 0 | 0 | BSV2-013 unfinished |
| [random/0 -C ×1.2](../../../outputs/bsbench-v2/results/evidence/8c5ee2a31992116d.json) | 2 | 2 | 0 | 1 | BSV2-013 unfinished,repetition |
| [random/3 +C ×0.8](../../../outputs/bsbench-v2/results/evidence/ae367fb838ad3200.json) | 1 | 1 | 0 | 1 | BSV2-011 unfinished,repetition |
| [random/4 -C ×1](../../../outputs/bsbench-v2/results/evidence/f76fbfba23c1795b.json) | 1 | 1 | 0 | 0 | BSV2-015 unfinished |
| [vjp_cache/0 -C ×1.2](../../../outputs/bsbench-v2/results/evidence/fa631b39c0ea779c.json) | 1 | 1 | 0 | 0 | BSV2-013 unfinished |
| [vjp_delta/0 -C ×1.2](../../../outputs/bsbench-v2/results/evidence/5ab15415726ea1c2.json) | 5 | 5 | 1 | 0 | BSV2-002 unfinished |

Three examples that become eligible: [mean_diff +C ×1.2, BSV2-014](../../../outputs/bsbench-v2/results/evidence/b09f18e251415702.json) has an empty steered answer (`""`); [random seed3 +C ×0.8, BSV2-011](../../../outputs/bsbench-v2/results/evidence/ae367fb838ad3200.json) ends in repeated zeros (`00000000000000000000…`); [vjp_cache −C ×1.2, BSV2-013](../../../outputs/bsbench-v2/results/evidence/fa631b39c0ea779c.json) ends `the actual physical limit of the setup rather` with no sentence ending. These failures stay visible in all saved raw evidence.

The ninth flagged evaluation dose, random seed4 −C ×1.2, has 9/20 role leaks and fails both predicates ([raw](../../../outputs/bsbench-v2/results/evidence/2f1d30b9289f4ac3.json)). `random` dose selection and rank under this counterfactual remain conditional on individual visibly broken answers being allowed in a cohort. The current selection is unchanged pending a scientific decision.

— PI/gpt-6-sol
