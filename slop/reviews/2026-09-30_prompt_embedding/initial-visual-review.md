## Verdict: pass with presentation limitations

Opened all three images before reading the complete report and regression log. No blocking clipping or gain-chart readability defect observed.

- **`outputs/bsbench/results/prompt-dev/prompt_gains.png`:** Higher embedding gain is not consistently stronger prompting. Short +C strongly increases premise acceptance at gains .125–2, reverses below bare at 4–8, and approaches bare at 16. Engineered +C remains strongly positive through 8, but those points fail admissibility. Both −C curves generally reduce premise acceptance. Lower damage at large gains does not establish useful reasoning or a breakdown boundary.
- The chart displays all nine tested gains for four series, including inadmissible points marked ×. The dotted damage cap and categorical spacing are explicit. Its caption correctly distinguishes persona sign from multiplier sign: “±C selects persona, not a negative gain.” Gain zero is explicitly not bare.
- Labels, legend, captions, and endpoints fit in the gain image without visible collisions. Overlapping series near gain zero and large-gain damage minima remain a minor reading limitation.

**Baseline distinction:** `plot.png` and `uat_plot.png` each visibly contain four ordinary-prompt stars, distinct from curve dots/rings/crosses. The gain chart uses no baseline stars. The default UAT view leaves both sweep pills disabled, so it does not visually demonstrate their enabled rendering.

**Nonblocking polish:** In `uat_plot.png`, left-side chars/VJP labels overlap, and right-side mean-difference/sink-split labels are crowded. `plot.png` places the positive prompt label very near the right edge, but its text and star remain visible. Also, × means *inadmissible* in the gain chart but *last admissible dose* in the Pareto chart; both explain this locally, though cross-figure readers could confuse them.

**Evidence:** `outputs/logs/prompt-baseline-star-regression.log` reports “prompt baseline stars=4; no swept-prompt stars or clipped baseline markers” and “UAT_PASS”; these are supplied results, not tests I reran. `slop/reviews/2026-09-30_prompt_embedding/results.md` appropriately limits interpretation to 20 dev questions, one seed, same-question dose selection, and no established breakdown boundary.

To remove the remaining regression uncertainty, inspect an enabled-sweep screenshot or assertion showing sweep curves present while baseline-star count remains four. The supplied `web/src/main.jsx` path was absent; implementation verification was therefore not completed.