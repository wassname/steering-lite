## Visual review

**Image successfully ingested.** Review was read-only; no jobs were run.

### What the image alone shows
Among the five displayed vector methods, VJP-resid achieves the largest leftward, control-adjusted pushback; VJP-value and topk_clusters show smaller shifts. All five move substantially rightward at stronger +C doses. Prompt −C sits close to zero adjusted premise change despite considerable off-axis change.

These are **effect–cost tradeoffs**, not evidence of cost-free steering: substantial premise changes generally accompany off-axis scores around 0.8–1.3. The image provides no uncertainty intervals for method comparisons and shows only selected methods.

The labels are largely readable at full resolution. The main ambiguity is the tightly clustered +C endpoint crosses near x≈1.9, where leader lines and pale-blue/green/blue markers are difficult to distinguish. Pale-blue and yellow-green labels have weak contrast. There is no major illegible text overlap, but the footnotes become small when embedded.

### What the grey lines mean
The caption correctly identifies dotted 10th/90th percentiles, dashed 25th/75th percentiles, and a solid median—not confidence intervals.

`random_zones` adds important qualifications:
- Each random direction contributes separate positive- and negative-sign walks.
- At each off-axis level, its effect is linearly interpolated at the walk’s **first crossing**.
- Quantiles use only walks reaching that level; plotting stops when fewer than half reach it, or at `MAX_OFF_AXIS`.
- Boundaries are subsequently smoothed.

Thus these are **conditional empirical effect quantiles among reaching walks**, not joint-density contours, probability-density levels, or regions containing a fixed proportion of complete trajectories. Both signs of a direction also are not independent directions. Being outside a grey boundary is descriptive, not a significance test.

### Are the strongest conclusions supported?
**Qualified yes:** the index supports VJP-resid’s strongest reported bidirectional score: −0.03, versus −0.22 for VJP-value and −0.94 for prompting. But VJP-resid’s 90% CI, [−0.17, +0.17], includes zero; separate method intervals do not establish pairwise significance.

The prompt comparison depends materially on the control penalty. Prompt −C gains **1.44 raw pushback**, versus VJP-resid’s **1.26**, but calls legitimate questions nonsense **48% versus 12%** (bare: 3%). Therefore “better selective pushback” is supported; “more raw pushback than prompting” is not. Random lacks this control adjustment, limiting like-for-like comparison.

**Off-axis magnitude is not coherence.** The index defines it as other differences from bare and admission as mean off-axis ≤1.5. Consequently, “last coherent dose,” “coherent doses,” and the code’s “coherence limit” overstate what this threshold establishes.

### Minimal changes
1. Replace “last coherent dose” with “last displayed admissible dose (mean off-axis ≤1.5), or before reversal”; fix corresponding index/docstring wording.
2. Add “first-crossing quantiles among reaching walks; not density contours; random unadjusted for controls.”
3. Mark the figure “selected methods”; describe upper corners as “low off-axis change,” rather than unqualified “clean.”
4. Separate the three rightmost endpoint labels/markers and darken pale text.

Limits: no raw-answer, judge-validity, or statistical-reproduction audit.