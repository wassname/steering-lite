# Final smoothing review: pure Pareto paths, all methods

Reviewer: Claude (reviewer-anthropic), read-only, fresh context. 2026-10-01.

## Observed (five PNGs + uat_plot.png)

- No return segments anywhere. Every solid/dashed path is monotone in on-axis effect: dev (7 methods), full, 27b-full, olmo-full, prompt-dev. Dominated last doses sit as isolated × (dev: pink (−0.88,0.79), orange (−0.53,0.84), olive (2.95,0.98); 27b: blue (−0.15,0.75), orange (−0.25,1.07), green (−0.42,1.13), green (4.93,0.85); olmo: green (0.12,0.88)). Non-dominated × are reached by the curve (full: olive (2.95,0.73), pink (−1.35,0.95), blue (−1.2,0.59)). Legend "× … not forced into curve" matches.
- Labels anchor at curve end: "prompt × gain +C" now points at (3.62,1.27); "VJP-resid +C" (27b) at the × (4.87,0.83). Legend reads "p90 ≈ 10th–90th".
- Bands: three nested smooth fills, flat bottom; 27b thin wedge; olmo narrow spike.

## Code (inference)

`smooth_path` refilters `frontier(include_endpoint=False)`, so the × enters only if Pareto; `directed(p) > ts[-1]` with Pareto ordering gives non-decreasing damage → PCHIP monotone. `uat.py:51` asserts this for every curve. `main.jsx:52` still hides fixed-grid labels in SVG (uat_plot.png has no "prompt × gain" text) — by design, not regression.

## Verdict: usable; no blocker. Bounded cosmetics

1. prompt-dev: brown eng −C dashed meets orange mean-diff −C dashed at the ring cluster (−0.85,0.2); reads as one path bare→(−1.75,0.77). Label at the far ring disambiguates.
2. dev: three long leaders ("linear_act −C", "mean difference −C", "eng. prompt × gain +C") cross the band; "mean difference +C" leader crosses the purple line. Known.
3. `smoothing-extent.log` max is 0.032 (27b p90, full p75), not the 0.011 in the handoff summary; results.md states it correctly. Per-dose interior shrink still unmeasured; global bound is small (<0.5% axis).
4. Vector supports with directed ≤ 0 would get dots but no line (`ts=[0.0]`, strict `>`). I saw none; check: count such supports in points.json.
5. Unresolved visually: eng −C first segment gain0→gain16 under the ring cluster; `uat.py:81` covers it.

```
 (\_/)  ¯\_(ツ)_/¯ "monotone at last"
 ( •_•) — reviewer-anthropic
```