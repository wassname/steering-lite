**Yes—the enabled-sweep concern is resolved. No new graphical blocker observed.**

Inspected all six actual images under `outputs/bsbench/results/prompt-dev/`.

- `uat_prompt_sweeps.png` visibly enables both sweep methods. Brown/purple curves use dots, rings and crosses rather than additional stars; crowded sweep endpoint labels are absent. The bare diamond remains isolated, without an invented origin-to-sweep segment.
- `scripts/bsbench/web/uat.py` explicitly asserts each fixed-grid path equals its measured support coordinates. The complete `outputs/logs/prompt-results-regression.log` reports “sweep-only curve marks=13; baseline stars still=4”, “UAT_PASS”, and “STATS_IDENTICAL: every method score, interval, room score, seed count and admissibility count”. These are supplied execution results, not independently rerun tests.
- `prompt_gains.png` remains legible, includes inadmissible tested doses, and explains sign, gain-zero and endpoint semantics. No clipping observed.
- `uat_selected.png` has readable, contained answer/judge text. `uat_full.png` shows continuous page layout; its reduced overview cannot establish fine-text readability.

**Residual polish:** Default `uat_plot.png` retains crowded chars/VJP and mean-difference/sink-split labels. In the sweep-only view, the ordinary +C prompt star overlaps nearby purple sweep markers; exact coincident results cannot be visually separated by position alone. Neither issue blocks interpretation with the separate gain chart and labeled baselines.