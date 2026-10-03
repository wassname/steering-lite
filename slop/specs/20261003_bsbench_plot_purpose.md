# BS-bench plot: purpose and design rules

Drafted by PI/OpenAI, edited by wassname, 2026-10-03; spelling fixed by PI/OpenAI. Every agent changing `scripts/bsbench/results.py` plot code or `web/src/main.jsx` reads this first. Marks without a stated job here should not be added.

## Purpose (wassname, quotes)


> "it was to take my steering-lite repo, and move it to a new eval and plot, as seen in vjp-steer. This gives us a bullshift bench steer, a judged eval, and a nice plot with baseline and random zone. it makes the pareto really clear. and incoherent sweep" (2026-09-24)

> "it's meant to describe the sweep from start to end, and the best tradeof is visually obvious as a pareto front. it doens't make sense to only draw to what yo uthink is best?" (2026-10-03)

> "the pareto best is and the end are the most important. I think pin at end? even if it bends back. if it's too noisy is just show we need to do more seeds or similar" (2026-10-03)

Purpose in wassname's words (2026-10-03):

> Steering can be thought of as "a dose of $CONCEPT steering, which changes the behaviour by X points, and has Y side effects, and eventually causes incoherence"
> When we steer a model, we want to change one thing without changing everything else. We might want less sycophancy, for example, while keeping its answers to ordinary factual questions the same. Give it pairs of prompts showing opposite behaviours, extract a steering vector, and apply it while the model generates. How well that works depends on the method and the strength of the steer.
> In this plot we are steering bluntness <> sycophancy on Bullshit Bench v2. So when we steer left we hope to see a reduction in sycophancy (x-axis) and when we steer right an increase. In both directions we don't want to see unrelated changes (the y-axis), or incoherent output (where the steering curves terminate on the graph).
> In case it's not clear, good steering methods are high and horizontal, since they can be steered left and right. Bad steering methods fall down as side effects accumulate, and then the line disappears as they fall off into incoherence (in the demos we see garbled and repeating text).
> The grey region is the null region where random vectors can steer the model, so any strong steering methods should be able to go outside this region before incoherence. Interestingly it's lopsided; this means it's easier to steer towards sycophancy than not. Many possible steering directions that occur in post-training show this effect where it's "downhill" towards the RLAIF direction, and "uphill" to avoid it.


Reference: `docs/vendor/vjp-steering/results/plot.png`, code `docs/vendor/vjp-steering/src/vjp_steering/results.py::plot`.

https://github.com/wassname/steering-lite

## The message (draft)

One sentence: for each method, turning the dose up moves the judged behaviour sideways (x: premise change, left = rejects the made-up premise, right = accepts it) and down (y: damage change); the further out a method gets before it drops into damage, the better. Eventually the steering dose gets so high that the model becomes incoherent, often repeating, changing topic, or gibberish. We use a judge to filter these out, so that we see a nice Pareto front which shows how the behaviour varies with dose, ending at the last coherent dose.

A good steering method should be better than a random intervention, and this null hypothesis is shown by the grey manifold, which shows how much effect random steering vectors achieve (outer contour: 10th to 90th percentile).

Steering should be compared to the most common steering method (mean difference, `mean_diff`), and to the most common alternative to steering (prompting), both of which are shown on the graph. A better steering method has a better Pareto-optimal point on both the right and the left, and beats random, the baseline, and prompting.

We see an asymmetry here, it's caused by two things: the base model often saturates the bench, being either very sycophantic or very blunt, making it hard to move in the saturated direction. Other times it resists changing behaviour in a direction; this is where steering often greatly exceeds prompting, not just in the lack of off-target effects (which we often achieve) but in a much stronger on-target behavioural change.

Questions a reader answers from the plot alone, in order:

1. Which methods get the best intended effect with the least side effects? (pareto frontier of outer envelope, read by eye)
   - And which method gets the greatest behaviour change before breaking down and disappearing from the chart?
2. Is that better than a random direction of the same size? (grey region)
3. Is it better than just asking with a prompt? (stars?, prompt sweeps?)
4. Where does each method stop being coherent, and does it bend back first? (end mark)

The table answers "how much better, with what uncertainty"; the plot does not need to carry the score.

## Elements and their jobs (draft; keep / drop to be agreed)

| Element | Job (question above) | Decision |
|---|---|---|
| Bare diamond at (0,0) | origin of every sweep | keep |
| Line per method and side, dose order, bare to last Jev-passing dose, median-filtered then spline (reference smoothing) | 1, 4: the sweep itself | keep; connect across a failed dose if a later dose passes |
| Solid dots on the line | measured doses vs smoothing | keep, small |
| x at the line end | 4: last coherent dose, "later doses rejected" | keep, pinned even if the line bends back |
| Grey random region, contoured (p90/p75/p50 shape) | 2 | keep; try without zero-fill (TODO) |
| Prompt stars | 3 | keep |
| Prompt gain sweeps as lines | 3 | keep; log-spaced gains from about 0.05 to 1 (TODO) |
| Ring at the score-setting dose | links plot to table | drop (reader finds the Pareto point by eye; table has the score) |
| Faint dots for passing doses off the line | none once lines show the whole sweep | drop by default; maybe on hover or when one method is shown in the web page |
| Line breaks at large effect gaps | none | drop (agent-invented) |
| Corner text "clean steer -> ...", "mostly side effects" | orientation | keep (reference) |
| Caption | define marks and the random region | keep short |

## Rules

- Jev mean damage <= 1.5 of 4 is the only coherence filter (see AGENTS.md). Doses that fail are not drawn on lines.
- A line connects passing doses in dose order, so it may bend back; that is information, not a bug.
- Display choices never change scores, selections or `points.json` measurements; check with a byte comparison when changing plot code.
- PNG and web page draw the same marks (`plot_marks.json` + `web/uat.py`).
- Read the tufte-viz skill and consider what earns its ink. View the web page yourself (Playwright or similar) and compare it to the reference, to catch regressions, text overlap, etc.
- Show the PNG to a fresh agent that has not seen this spec or the code; it must recover the message above. If it misreads the plot, fix the plot, not the agent (wassname 2026-10-03: a blind model "totally misunderstood" the current plot).

## Decisions (wassname, 2026-10-03)

- Smoothing: "keep, it looks nice" (reference median filter + spline).
- Ring: "I guess remove, the person can find the pareto optimal we don't need to clutter."
- Failed dose followed by passing doses: "yes connect across failed".
- Faint dots: "perhaps they could show up on mouse over, or if only one line visible on the html if it's not too complex ... it's too much information by default".
