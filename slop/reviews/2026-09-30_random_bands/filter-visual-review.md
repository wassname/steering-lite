## Final assessment

No filtering or scientific-definition blocker found. One screenshot-evidence defect remains.

**Rejected-dose leakage:** none found in the default plots. In `scripts/bsbench/results.py`, `prompt_gain_plot` obtains accepted gains from `method_curve`, substitutes `None` otherwise, and sets `connectgaps=False`. Pareto supports use the same production filter; ordinary stars require `admissible` in both renderers. Rejected doses appear only in the explicitly labelled diagnostic plot, hidden inside closed React details.

The supplied `filter-probe.log` reports: “one failed seed excludes the whole gain; accepted next gain preserved; failed gains not bridged”. This supports multi-seed filtering without inventing a permanent boundary. Gain 16 can recover. `production-run.log` reports “PRODUCTION_STATS_IDENTICAL: scores, intervals, selection, points and curves”. These are supplied execution results, not independently rerun checks.

**Code:** the change reuses existing acceptance/frontier logic, adds one diagnostic flag rather than a second plotting implementation, and includes both methods in `justfile` sweep defaults. Scope and size are reasonable.

**Visual observations:** inspected all five latest Pareto PNGs, default/diagnostic gain PNGs in dev and prompt-dev, and current browser screenshots. The title collision is resolved. Exported quantile definitions and “not confidence or sample-coverage regions” are legible outside the plotting area. Nested shades remain visible; 27b’s narrow region is naturally harder to distinguish. Dev is crowded but usable. Default gain plots visibly omit rejected gains and preserve gaps. The browser gain section explicitly warns: “A passing dose can still contain damaged answers.”

**Remaining evidence issue:** `outputs/bsbench/results/dev/uat_plot.png` and `outputs/bsbench/results/prompt-dev/uat_plot.png` show the scrolled gain section, not the opening comparison. In `scripts/bsbench/web/uat.py`, the gain-section screenshot precedes the viewport screenshot; scrolling caused by that capture is the likely explanation. Restore scroll-to-top and regenerate. A fresh screenshot showing the selected methods and complete opening plot would resolve this. Source, static figures and UAT assertions support the requested opening selection, but these particular screenshots do not demonstrate it.

— PI/OpenAI, independent visual/definition reviewer