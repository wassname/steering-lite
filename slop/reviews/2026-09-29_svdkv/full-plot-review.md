## Substantive findings

**No demonstrated blocker.** Two interpretation risks remain:

- `assets/bsbench_qwen3.5-4b_full.png` labels the grey envelope “null zone of random directions” without defining its construction or coverage. Readers could mistake being outside it for statistical significance. Neither `README.md` Results nor `outputs/bsbench/results/full/index.md` supplies that definition. This affects comparisons with random; inspect the envelope construction to determine whether a confidence-region interpretation is justified.
- The figure displays all four prompt stars, but the full index omits inadmissible prompt directions, and README says one direction “failed the coherence or damage check.” The image alone does not identify those failures. Mark inadmissible stars explicitly; checking their underlying eligibility would distinguish a deliberate reference display from accidental inclusion.

## Image-first interpretation

Before reading the documents: horizontal displacement measures signed premise acceptance; upward means less damage. Green VJP-value reaches furthest left among plotted methods. Blue VJP-resid and green look strong toward +C. Prompt baselines incur substantially greater damage. Rings identify score-setting doses, whereas crosses identify final coherent doses: endpoint ordering is not score ordering.

## Cross-checks

The displayed five match the full-index point-estimate ranking: VJP-value, chars, linear_act, sink_split_resid, VJP-resid. Sink’s −C ring near (−0.92, 0.21), rather than its cross near (−0.47, 0.42), supports its reported score around 0.70. README correctly places it fourth.

Sink is consistently mustard/gold in `outputs/bsbench/results/dev/plot.png` and the full asset; dev’s orange method is mean difference, not sink.

No obvious clipping. Labels remain inside the canvas. The paired sink-versus-mean_diff claim cannot be validated from these images or marginal intervals; its linked analysis was outside this review.

## Cosmetic suggestions

Separate the overlapping pink/gold −C rings slightly through annotation, not coordinate changes. Add “best on-axis minus off-axis dose, per side” beside the ring legend.