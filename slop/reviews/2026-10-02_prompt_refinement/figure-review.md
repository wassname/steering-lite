Everything needed is verified. Final artifact follows.

---

# Figure review (final, caption follow-up): prompt refinement — dev, prompt-dev, full, 27b-full, olmo-full

Reviewer: PI/Anthropic. Read-only. This replaces the previous version of this file. Scope of this pass: confirm the P1 caption fix landed in produced artifacts, caveats are visible, and no new figure blocker appeared. Geometry was not re-audited; `caption-verification.log` states the data did not change.

## Evidence (observed)

- `caption-verification.log`: `CAPTION_DATA_IDENTICAL` for all five cohorts — "points, scores, intervals, selected doses, raw bounds and plot support byte-identical".
- `results.py:488` footer now reads: `p90/p75 target 10–90%/25–75%; p50 = median<br>discrete ranks; small samples can span min–max; smoothed, zero-filled; not confidence/coverage regions; may be asymmetric`.
- All five `plot.png` re-ingested; each footer shows exactly that text with the cohort's own count (dev 32, prompt-dev 32, full 11, 27B 8, OLMo 3). Four footer lines fit inside the canvas in every image; no clipping. Marks, rings, ×, labels and envelopes are visually unchanged from my prior pass.
- `main.jsx:188` (built page text): "Bounds use discrete observed ranks: with few samples, p90 can be min–max (for example, OLMo's three directions give six signed values). Eligible counts vary by dose; see the table below." and `main.jsx:186`: "The ring marks the strength used for the score, even if the effect is near zero or in the wrong direction. Checks use cohort means; individual retained answers can still be badly damaged."
- `uat.py:50–51` now asserts both strings in the rendered body. `caption-browser-four.log`: 4× `UAT_PASS` (13:38–13:39). `caption-browser-final.log`: 5× `UAT_PASS` (lines 25/50/72/94/116; last 13:44:42). `coverage.log` (13:46:37): `HASH_PASS: 568 historical answer/full-certificate files unchanged`, `COVERAGE_PASS` rows present (file was mid-write on my first read; second read succeeded).

## P1 status: resolved

The earlier P1 was that "p90 ≈ 10th–90th" was stated identically for OLMo (6 pooled values → min–max) and 27B (16 values → 2nd–15th). The PNG footer now says "target" and "discrete ranks; small samples can span min–max"; the web page names the OLMo case and points to the per-dose eligibility table. That is the wording fix I asked for; no threshold was added and the envelope data is unchanged. Resolved.

## Other caveats now visible in artifacts

- Wrong-side / near-zero rings (dev eng +C; OLMo VJP-value −C): explicit in `main.jsx:186`. PNG footer still does not say it — acceptable, the ring semantic is unchanged and the page is the primary reader surface.
- Passing mean ≠ per-answer coherence: `main.jsx:186` and the gain-sweep section. Asserted by `uat.py:51`.
- Low-gain persona-indistinguishability (prompt view lede) retained from the prior rebuild.

## No new blockers

No new geometry, label, clipping or count issue in the five current PNGs. Previously reported P2 cosmetics stand unchanged and are not re-litigated: leader lines crossing label text in the web SVG, symmetric x-range driven by one outlier (OLMo, 27B), PNG vs SVG x-limit inputs differing (`results.py:424` vs `svg_labels()`), eng +C gain-chart series nearly invisible.

## Verdict

All five cohorts: **usable** as current evidence. P1 withdrawn as resolved. Prior NOT_READY finding remains withdrawn. No held-out or multi-seed prompt claim is supported.

Inspect: `/workspace/2026/lite/steering-lite-bsbench/outputs/bsbench/results/olmo-full/plot.png`, `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/results.py:488`, `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/web/src/main.jsx:186`, `/workspace/2026/lite/steering-lite-bsbench/scripts/bsbench/web/uat.py:50`, `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-02_prompt_refinement/caption-browser-final.log:116`.

— PI/Anthropic