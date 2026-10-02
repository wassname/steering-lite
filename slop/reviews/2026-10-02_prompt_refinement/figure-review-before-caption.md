All evidence is in. Here is the replacement artifact.

---

# Figure review (final): prompt refinement — dev, prompt-dev, full, 27b-full, olmo-full

Reviewer: PI/Anthropic. Read-only fresh-eyes pass on the rebuilt figures. This replaces my earlier NOT_READY verdict for the three full reports, which was issued mid-rebuild.

## Freshness (observed)

- `final-render.log` contains five `UAT_PASS` lines (dev 12:53, prompt-dev 12:55, full 13:02, 27b-full 13:05, olmo-full 13:06).
- `points.json` has `"plot_gap": 1.0` in all five (`full:10`, `27b-full:9`, `olmo-full:9`, `dev:12`, `prompt-dev:8`); `plot_marks.json` carries `passing_marks` in all five (full 8, 27B 14, OLMo 35, dev 65, prompt-dev 37).
- `comparison.log`: `FULL_SCIENTIFIC_REGRESSION_PASS` ×3, `HISTORICAL_BOOTSTRAP_REPLAY_PASS` for 27b-full and olmo-full. I did not re-derive the RNG-order explanation; I accept the replay evidence as supplied.
- Legend counts match the JSON: dev/prompt-dev 32, full 11, 27B 8, OLMo 3, in both PNG and SVG headers.

## What each plot says (from the images)

**Axes, all main plots.** x = judge-rated change in premise acceptance vs the bare answer (left = rejects the false premise more; right = goes along with it). y = mean |change| in a 0–4 damage rating, 0 at top. Lines are Pareto supports in effect order, not dose paths. Ring = score-setting dose, × = last passing dose, faint dot = dominated passing dose, grey = pooled-sign percentile envelope of random directions at median damage.

**dev (4B, 20 q).** VJP-value and chars −C reach ≈−1.75 at ≈0.33 damage, far left of the envelope (which barely passes −0.2 at that damage). On +C most methods ride the envelope's right edge to ≈+1.8, then exit it at 0.65–1.1 damage. The engineered −C prompt star (−1.53, 1.09) is dominated by the two steering rings. Prompt +C star (3.57, 1.22) is the most sycophantic and the most damaged point.

**prompt-dev.** Same axes, sweeps + mean difference. The +C sweep's support is three disconnected marks: near-bare cluster, a lone dot at (1.75, 1.08) = gain 3.75, and the ring/star at (3.57, 1.22) = gain 1; both spans >1 are honestly unbridged. The −C sweep ring at (−0.84, 0.16) is gain 1/256; the web lede now says "Low-gain responses can be similar across personas; scores do not establish instruction-specific steering", and `comparison.log SELECTED_CONTROLS` confirms it (−0.841 vs opposite persona −0.707). Eng +C remains a single ring left of bare.

**Gain charts.** All 31 static ticks are present and legible (0 … 0.000976562 … 3.75, 4, 6, 8, 12, 16); the "hover" wording is gone. Admissible chart: both personas sit at −0.3…−0.85 for gains ≤1/32 and again at −0.25…−0.6 for gains ≥6; +C passes at 0.5/0.75/1/1.5 right at the 1.5 cap, then 3.75 (+1.75) → 4 (−0.9) adjacent and bridged correctly. Diagnostic chart shows the ×'s filling every gap.

**full (4B, 100 q, 11 random).** −C: VJP-value (−1.6, 0.46), chars (−1.43, 0.55), VJP-resid × (−1.2, 0.59), linear_act × (−1.35, 0.95) — all left of the envelope. +C: four methods converge at +2.5–2.9 with 0.45–0.55 damage while the p90 envelope reaches ≈+2.3 at its 0.61 cutoff; the +C advantage over random is visibly small. Only 8 faint dots exist.

**27b-full (8 random).** Envelope is a thin wedge to (1.4, 0.24). +C: VJP-resid/VJP-value reach +4.8–4.9 at ≈0.6; mean_diff ring at (5.4, 1.17). Three large +C gaps (2.1→3.2, 2.4→4.2, 2.8→5.4) are shown as breaks, not lines. −C is weak (best ring −1.23 at 0.32); the −C ×'s sit near x≈0 at 0.75–1.1 damage, i.e. the strongest passing doses lose the effect.

**olmo-full (3 random).** Everything within ±1.7 except the engineered −C star at (−3.11, 0.84), which beats every steering method. Four of eight rings sit within ±0.3 of bare (VJP-value −C at +0.02). Envelope is a narrow vertical wedge to 0.82.

## Checks

- No return segments: supports are monotone by construction (`frontier()`), `uat.py` asserts "Pareto path must not double back", 5× pass. Visual scan agrees.
- Gap honesty: every unsupported span >1 I could measure is broken (dev 1.07/1.2; 27B 1.1/1.8/2.6); 0.89 is bridged. Consistent with `MAX_PLOT_GAP`.
- Clipping: `uat.py:46` asserts default-view labels inside the SVG bbox; no PNG label or mark touches an edge in any cohort (closest: 27B mean_diff +C ring, OLMo eng-prompt star).
- SVG/PNG default view: same methods (`plot_marks.methods == shown`), same frontier/passing counts, same label set (`svg_labels()` reuses `place_labels`). Remaining differences are tick values (PNG round, SVG eighths) and label offsets.
- PNG y-range now includes envelope height (`results.py:425–426`).

## Remaining issues

**P1 (potentially misleading, not cosmetic).** Legend text "p90 ≈ 10th–90th, p75 ≈ 25th–75th" is identical across cohorts, but with OLMo's 3 directions (6 pooled values) `tail = 6*10//100 = 0`, so "p90" is min–max and "p75" is 2nd–5th; with 27B's 16 values p90 is 2nd–15th. The envelope is still empirical, but a reader comparing OLMo's wedge to dev's band will assume equal percentile meaning. Disprove by: `random_zones()` tail arithmetic at n=6 (`results.py:357`). Fix is wording (state n or drop "≈ 10th–90th" when n<10), not a new filter.

**P2 (honest but easy to misread).** (a) Rings at near-zero or wrong-side effect (dev eng +C; OLMo VJP-value −C at +0.02) look like "a result"; the prompt-view lede covers only the prompt case. (b) Main-plot legend does not say that a passing mean can contain damage-4 answers; the web gain section does. (c) The gain chart's eng +C series is two isolated dots, one hidden under three others at gain 0.

**P2 (cosmetic).** Web leader lines strike through other labels ("eng. prompt ×/gain -C" in dev and prompt-dev `uat_plot.png`) because `svg_labels()` passes only points, not path samples, as obstacles. Symmetric x-range driven by one outlier leaves half the canvas empty (OLMo, 27B). PNG `x_limit` includes individual random points (`results.py:424`) while the SVG does not; equal today, could diverge.

## Verdict

All five cohorts: **usable** as current evidence for internal review, with the P1 legend caveat attached to the 27B and OLMo envelopes. The earlier NOT_READY finding is withdrawn. No held-out or multi-seed prompt claim is supported (prompt sweeps are seed 0 only).

Inspect: `/workspace/2026/lite/steering-lite-bsbench/outputs/bsbench/results/olmo-full/plot.png`, `/workspace/2026/lite/steering-lite-bsbench/outputs/bsbench/results/27b-full/plot.png`, `/workspace/2026/lite/steering-lite-bsbench/outputs/bsbench/results/prompt-dev/prompt_gains.png`, `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/results.py:357`, `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/results.py:575`.

— PI/Anthropic