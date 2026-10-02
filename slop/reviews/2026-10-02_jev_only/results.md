# Jev-only coherence for plotted and scored doses

PI/OpenAI, 2026-10-02. Branch: dev/prompt-gains-random-reference.

User correction:
> only thje judge is reliable so fa,r emchanial checks are just for initial calib

## Change

- Jev mean steered damage ≤1.5/4 is the sole coherence criterion for plots and scores.
- Mechanical flags and mechanical walk boundaries remain recorded diagnostics; neither rejects a judged point.
- Seed/question coverage requirements remain. Existing walks were bounded by mechanical calibration checks, so ungenerated doses cannot be recovered by re-filtering.
- Do not change the damage threshold or cached aware ratings. No new generation. Refresh only missing blind ratings for newly selected doses, then rebuild normal reports.

## Gain 12 and 16 verification, short sycophantic instruction

Source: per-question Jev ratings in the existing prompt-dev points.json (prior filter snapshot).

| Gain | Mean Jev damage /4 | Maximum individual damage /4 | Questions above 1.5 | Premise change |
|---|---:|---:|---:|---:|
| 12 | 0.2635 | 1.29 | 0/20 | -0.673 |
| 16 | 0.2040 | 0.79 | 0/20 | -0.105 |

Thus these gains passed Jev, not merely mechanical checks. This does not show stronger sycophantic instruction following: ordinary gain 1 has premise change +3.5705. One retained answer echoes the question and then answers; Jev damage is 0.79. Passing the rating threshold is not a guarantee of factual correctness or ideal output.

## Verification

Five normal `results.py` rebuilds, then Vite + browser UAT for all five reports. `verify.py` compares each report with the pre-change snapshot `.local/jev-only-before/` (log: `verification.log`).

- 568 historical 4B answer/certificate files byte-identical. 27B/OLMo have no byte hashes; their aware Jev cache checks show `missing=0`, so no answer text changed.
- Same `(method, seed, side, C)` keys in every report; every point field and per-question rating unchanged except `admissible` and newly attached blind ratings.
- Every point: `admissible == (steered_damage <= 1.5)`.
- Judging: no aware re-rating. 100 new blind ratings on 27B, $0.0059 (random seed 6, +C, C=10.08, now inside a Pareto-best dose).

Newly admitted points (all failed only a mechanical check; none post-boundary):

| Report | Method | Seed | Side | C | Mean Jev damage | Mechanical flag |
|---|---|---:|---|---:|---:|---|
| dev, prompt-dev | corda_pca | 0 | −C | 25.40 | 1.377 | role_leak |
| dev, prompt-dev | value_gram | 0 | −C | 6.35 | 1.3425 | repetition |
| full 4B | sspace_pool | 0 | −C | 6.35 | 1.2319 | repetition |
| full 4B | sspace_pool | 1 | −C | 6.35 | 1.4287 | repetition |
| full 4B | value_gram | 0 | −C | 6.35 | 1.3819 | repetition |
| full 27B | random | 6 | +C | 10.08 | 1.0625 | role_leak |
| full OLMo | — | | | | | none |

Summary changes: **no method score changed.** N/rejected counts shift for corda_pca, value_gram (dev, full), sspace_pool (full). The 27B random +C best-dose effect drops 3.96 → 3.10 at the same C (pooled mean now includes seed 6; off-axis .761 → .788).

Limits: existing walks stopped at mechanical boundaries, so doses past them were never generated; re-filtering cannot recover them. Mean damage ≤1.5 does not certify each answer (27B admissible doses contain individual answers rated 3.5).

Independent review (Anthropic, fresh context): `review.md`. One blocker: the prompt gain section still said "healthy answers, not past a walk boundary". Fixed in `main.jsx`, UAT now asserts the Jev-only wording, all five rebuilt (`browser-wording.log`, `verification-final.log`).

-- PI/OpenAI
