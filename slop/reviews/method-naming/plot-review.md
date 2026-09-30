# Plot review — method naming (visual check only)

**What they show.** All four are the same layout: x = judge on-axis change in premise level (solid +C, dashed −C), y = off-axis damage (lower better), black diamond = bare, grey blob = null zone of random directions, rings = score-setting dose, × = last coherent dose, ★ = prompt baselines. Method labels now read `VJP-value`, `VJP-resid`, `chars`, `mean difference`, `linear_act`, `spherical`, `sink_split_resid`, `prompt`, `eng. prompt`, each suffixed ±C. Labels are legible and colour-matched; every ring/× has a leader line, so mapping is followable.

**Concrete defects observed.**

1. **qwen3.5-4b_full:** the `null zone of random directions` annotation is overprinted by the `VJP-value +C` leader line and the VJP-value/VJP-resid rings; "random directions" text is partly obscured. Also `VJP-value +C` and `VJP-resid +C` rings overlap almost exactly (~x=2.6–2.75, y≈0.45), so ring ownership must be read from the leader lines.
2. **qwen3.5-27b_full:** `null zone of random directions` annotation is overstruck by the Pareto dots near bare ("random directions" partly covered). `VJP-value +C` and `VJP-resid +C` rings and × markers overlap at x≈4.8; the `VJP-value +C` leader line crosses the VJP-resid curve. Also `prompt +C` label collides with its own star glyph (text touches marker).
3. **olmo-2-32b_full:** `null zone of random directions` annotation is drawn over the orange/cyan dashed curves and the grey blob, reducing legibility. Rings at x≈0/−0.1 (VJP-value +C and VJP-resid +C) overlap.
4. **dev/plot.png:** `null zone of random directions` overlaps the `VJP-value +C` label ("VJP-value +C" sits on the last line of the annotation). `sink_split_resid +C` and `mean difference +C` × markers sit ~2 px apart at x≈2.95–3.1 with crossing leader lines.

**No clipping** at the canvas edge in any file; naming is consistent across the four plots. Inconsistency note only: `mean difference` (space) vs `linear_act`/`sink_split_resid` (underscores) vs `VJP-value` (hyphen) — mixed styles, not a readability defect.