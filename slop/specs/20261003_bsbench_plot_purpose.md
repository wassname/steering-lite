# BS-bench plot: purpose and design rules

DRAFT by PI/OpenAI, 2026-10-03, for wassname to edit. Every agent changing `scripts/bsbench/results.py` plot code or `web/src/main.jsx` reads this first. Marks without a stated job here should not be added.

## Purpose (wassname, quotes)

> "it was to take my steering-lite repo, and move it to a new eval and plot, as seen in vjp-steer. This gives us a bullshift bench steer, a judged eval, and a nice plot with baseline and random zone. it makes the pareto really clear. and incoherent sweep" (2026-09-24)

> "it's mean to describe the sweep from start to end, and the best tradeof is visually obvious as a pareto front. it doens't make sense to only draw to what yo uthink is best?" (2026-10-03)

> "the pareto best is and the end are the most important. I think pin at end? even if it bends back. if it's too noisy is just show we need to do more seeds or similar" (2026-10-03)

Reference: `docs/vendor/vjp-steering/results/plot.png`, code `docs/vendor/vjp-steering/src/vjp_steering/results.py::plot`.

## The message (draft)

One sentence: for each method, turning the dose up moves the judged behaviour sideways (x: premise change, left = rejects the made-up premise, right = accepts it) and down (y: damage change); the further out a method gets before it drops into damage, the better, and the grey manifold shows what a random direction does at the same dose.

Questions a reader answers from the plot alone, in order:

1. Which methods move furthest in each direction while staying near the top? (outer envelope, read by eye)
2. Is that better than a random direction of the same size? (grey region)
3. Is it better than just asking with a prompt? (stars, prompt sweeps)
4. Where does each method stop being coherent, and does it bend back first? (end mark)

The table answers "how much better, with what uncertainty"; the plot does not need to carry the score.

## Elements and their jobs (draft; keep / drop to be agreed)

| Element | Job (question above) | Proposal |
|---|---|---|
| Bare diamond at (0,0) | origin of every sweep | keep |
| Line per method and side, dose order, bare to last Jev-passing dose, smoothed | 1, 4: the sweep itself | keep; was Pareto-only, change back |
| Solid dots on the line | measured doses vs smoothing | keep, small |
| x at the line end | 4: last coherent dose, "later doses rejected" | keep, pinned even if the line bends back |
| Grey random region, contoured (p90/p75/p50 shape) | 2 | keep (wassname likes the manifold shape); fix the edge at x=0 from zero-fill |
| Prompt stars | 3 | keep |
| Prompt gain sweeps as lines | 3 | keep; gain grid 0.05..1 (TODO) |
| Ring at the score-setting dose | links plot to table | undecided: redundant with the table and the visible envelope |
| Faint dots for passing doses off the line | none once lines show the whole sweep | drop (they disappear: every passing dose is on its line) |
| Line breaks at large effect gaps | none | dropped (agent-invented) |
| Corner text "clean steer -> ...", "mostly side effects" | orientation | keep (reference) |
| Caption | define marks and the random region | keep short |

## Rules

- Jev mean damage <= 1.5 of 4 is the only coherence filter (see AGENTS.md). Doses that fail are not drawn on lines.
- A line connects passing doses in dose order, so it may bend back; that is information, not a bug.
- Display choices never change scores, selections or `points.json` measurements; check with a byte comparison when changing plot code.
- PNG and web page draw the same marks (`plot_marks.json` + `web/uat.py`).

## Open questions for wassname

- Smoothing: the reference median-filters points over neighbouring doses (log-C window) then draws a spline. Keep raw points with a spline, or copy the median filter?
- Ring: keep or drop?
- Doses passing after a failure (fixed prompt grid, some walks): connect across the failed dose (reference behaviour) or end at the first failure?
