I have what I need. Here is the review artifact.

---

# Figure review: prompt refinement (dev / prompt-dev / full×3)

Reviewer: PI/Anthropic (fresh eyes, read-only). Scope as assigned. Observation vs inference marked.

## What the figures say (plain language)

**Main plot** (`plot.png`, web SVG): each point is one dose of one method, averaged over 20 questions. x = how much the judge says the answer's acceptance of the false premise moved vs the bare answer (left = rejects more, the useful direction for −C; right = goes along, the target for +C). y = mean absolute change in a 0–4 "damage" rating (rambling, filler, off-topic), 0 at top. Lines join a method's non-dominated doses in *effect order* — a Pareto support, not a dose path. Rings mark the dose used for the score; × is the last passing dose; faint dots are passing doses that are dominated. Grey bands are percentile envelopes of 32 random directions (both signs pooled), drawn at the median damage of those directions.

**Gain charts**: x = multiplier on the instruction-token embeddings (0 = zeroed tokens with positions kept, 1 = ordinary prompting), left panel = premise change, right = absolute mean damage against the 1.5 cap.

## Freshness

- Observed: `prompt-dev/points.json:8` and `dev/points.json:12` → `"plot_gap": 1.0`; both list `random_seeds` 0–31 (32). `dev/plot_marks.json` → `{"frontier_marks": 77, "passing_marks": 65, ...}`. `dev-render.log` ends `UAT_PASS` for both. **Current.**
- Observed: `full/`, `27b-full/`, `olmo-full/points.json` have no `plot_gap` key; their `plot_marks.json` lack `passing_marks`; `full/plot.png` legend reads the old "dot = Pareto point · ring = dose that sets the score … not confidence or sample-coverage regions" and shows no faint dots. `full-regression.log` stops at `12:20:02 … exclude random_s15_full.json`. **NOT_READY — stale; do not cite.**

## Geometry checks (dev, prompt-dev)

- Returns/backward curves: none. `frontier()` (results.py:377) keeps non-dominated points so damage is non-decreasing along a support; `smooth_path()` is a monotone cubic; `uat.py` asserts "Pareto path must not double back". Verified against data: `sink_split_resid +C` support 0.56→0.66→1.01→1.31→1.81→3.01, damage 0.104→0.819 (dev/points.json:17094). My first impression of an olive backward step was wrong.
- Gap rule: `sink_split_resid +C` 1.81→3.01 (1.2) and `vjp_value −C` 0.70→1.78 (1.07) are correctly left unbridged; `eng. prompt −C` 0.85→1.74 (0.89) is bridged.
- Faint dots: PNG `other` = curve minus support; web `circle.sample` same; `uat.py` asserts `circle.sample == passing_marks == other`. Equivalent.
- Cap: `admissible` requires mean steered damage ≤ 1.5 (results.py:97); bootstrap reapplies it. Observed the +C prompt sits on a knife-edge: gains 0.5/0.75/1/1.5 pass at ≈1.47, 1.25 and 1.75–3.5 fail (uat_prompt_gains table).
- Random envelopes: `random_zones()` uses sorted pooled effects with integer tail index (hence "≈"), zero-filled, Chaikin-smoothed with shared weights so nesting holds; asymmetry is purely empirical (dev band spans ≈ −0.2…+2.9). Legend text says "not confidence/coverage". Correct.

## Issues

**P0 (process)** — Three full reports are stale (above). Any full-cohort claim is unsupported until rebuilt and `UAT_PASS` logged.

**P1 (interpretation, prompt×gain)** — Observed in `prompt_gains.png`/`_all.png` and dev/points.json:15017: both +C and −C sweeps move premise to −0.4…−0.85 at gains 0–0.03 with low damage, and again at gains 8–16 (−0.25…−0.6). `prompting_scale −C`'s score-setting dose is gain 0.0039 (−0.841, 0.16) and `prompting_engineered_scale +C`'s is gain 0 (−0.36). Inference (likely, ~65%): these are null-token/position artefacts, persona-independent — supported by the blind judge at eng +C gain 0: "different_advice 19%, technical 15%" with P(intended) 6%. The −C "prompt × gain" rank (7th, +0.68) and the eng +C ring drawn *left* of bare under a "+C" label invite the wrong reading. Disprove by: comparing +C vs −C per-question effects at gain ≤0.03 — if persona matters, signs should differ; the chart shows they do not.

**P1 (readability, gain PNG)** — Categorical x with only 9 of ~40 ticks labelled and "hover for every gain" in a static PNG; the 3.75→4 flip (+1.75 → −0.9) cannot be located without the table.

**P1 (web)** — `main.jsx` places labels at `path.at(-1)` with no collision avoidance: `uat_plot.png` shows "lin"/"mean_di" overlapping. Fixed-grid methods get no label at all, so in the prompt view the two swept lines — the page's subject — are unlabelled.

**P2** — (a) Two rings coincide at (−0.85, 0.15): mean_diff −C (−0.86) and prompt×gain −C (−0.841); purple hidden under orange. (b) For fixed-grid sweeps the × (gain 16) sits near bare at the *start* of the support, opposite to walks. (c) Unlabelled lone Pareto dot at (1.75, 1.08) (prompt +C gain 3.75). (d) Flat envelope bottom at ≈0.71 is the half-seeds cutoff, not a damage ceiling; unannotated. (e) Main-plot y is mean |Δdamage| while the cap is absolute mean — legend doesn't say; a passing mean can hide damage-4 answers. (f) PNG y-range ignores zone extent (results.py:425); web includes it — could diverge on other cohorts.

## Verdict

dev and prompt-dev figures: **usable** for internal review, with the P1 prompt×gain caveat stated wherever the sweep is discussed. full/27b-full/olmo-full: **not usable** (stale). No held-out or multi-seed prompt claim is supported by these artefacts (prompt sweeps are seed=1).

Inspect: `/workspace/2026/lite/steering-lite-bsbench/outputs/bsbench/results/prompt-dev/prompt_gains_all.png`, `/workspace/2026/lite/steering-lite-bsbench/outputs/bsbench/results/prompt-dev/uat_plot.png`, `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/results.py:377`, `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/web/src/main.jsx:46`.

— PI/Anthropic