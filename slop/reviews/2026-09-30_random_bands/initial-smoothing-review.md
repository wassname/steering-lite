# Smoothing review: random bands + prompt Pareto paths

Reviewer: Claude (reviewer-anthropic), read-only. 2026-10-01.

## What I see (observed)

- `prompt-dev/plot.png`, `dev/plot.png`: one solid + one dashed purple line. Solid purple runs from gain 0 (≈−0.47, 0.13) through the +C gain-16 cross (−0.1, 0.15) in one smooth arc to the star/ring (3.57, 1.22) and a short tail to (3.62, 1.27). Dashed purple leaves the same gain-0 point leftward through the ring (−0.62, 0.15) and plunges to (−0.88, 1.12); the −C gain-16 cross (−0.22, 0.15) sits alone. Matches `curve-order.log` (`+C [0,16,1,.5]`, `-C [0,8,4]`). The old two-segment return is gone. Brown eng +C is a lone ring at gain 0 (−0.35, 0.1) with an isolated cross (−0.58, 0.24); eng −C runs cross→ring (−1.75, 0.77).
- Grey bands: three nested smooth fills, flat bottom at the last support, titles say "dev, 20 questions" / "full, 100 questions"; web SVG shows "DEV · 20 questions". `prompt_gains.png` has gaps at 0.125/0.25/2 and no bridging lines.
- `full`, `27b-full`, `olmo-full`: curves unchanged in composition; bands smooth; 27b band is a thin wedge above damage 0.25.

## Code check (observed → inference)

`frontier(include_endpoint=False)` inside `smooth_path` for fixed grids is a correct Pareto set (dominance is transitive, so double-filtering equals one pass). Pareto order implies non-decreasing damage, so PCHIP is monotone; no bare anchor; one trace per (method, side). Chaikin on shared `(damage, lo, hi)` tuples with positive weights preserves `lo90≤lo75≤lo50≤0≤hi50≤hi75≤hi90` pointwise and keeps `bounds` raw; `render_saved.py` asserts both. `data.py:14 RANDOM_SEEDS = range(11)` pins production, so the 16-seed probe cannot leak in. Probe claims check against `reference-probe.json`: max p90 Δ = 0.099 (C=2 right), median-damage Δ ≤ 0.005, scoped to full-100. Not overreaching.

## Verdict: usable, no blocker. Residual risks

1. **Return segments remain for vector walks**: mean diff −C ring (−0.86, 0.19) → cross (−0.53, 0.84) in prompt-dev/dev; sink_split_resid −C in full; all four −C plus VJP-value/resid +C in 27b-full. Same visual pattern the user flagged, now orange/green/blue. Legend implies it ("no forced return for prompt gains"). Highly likely the user reads it as the same bug; ask whether to isolate × for all methods.
2. Fixed-grid labels anchor at `curve[-1]` (gain-16 cross), so "prompt × gain +C" sits near bare while its arc ends at (3.6, 1.27). Web (`main.jsx`) suppresses fixed-grid labels entirely.
3. Both ±C prompt paths start at gain 0; the solid +C line lies left of bare. Correct per rule, but a fold-like junction persists in the purple cluster.
4. eng +C score-setting dose is gain 0 (erased prompt): every positive-effect gain failed the cap. Worth a sentence in results.
5. Chaikin cuts interior convex vertices: band reach at interior doses slightly under the measured percentile (≈0.01 on full p90-left at C=0.79, from probe numbers; dev unquantified). Check: `max(path x)` vs `max(bounds hi)` per zone. PNG legend says "10th–90th" without "roughly" (22 values → `effects[2]`/`effects[-3]`).
6. Missing evidence: producer of `curve-order.log` not in bundle (uat.py covers the same monotone assertion); I could not visually resolve the short solid segment gain 0→16 under the cross cluster.

```
  (\_/)   🔍 "no shell: fortune not drawn"
  ( •_•)  — reviewer-anthropic
```