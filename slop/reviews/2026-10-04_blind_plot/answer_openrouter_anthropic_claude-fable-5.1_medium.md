[pi-web-access] Dynamic tool activation requires Pi 0.86.1 or newer; web tools remain eagerly available.
**1. What each chart measures**

*Chart 1 (plot.png)*
- x = how far the model's "premise level" moves from the unsteered model (left = more skeptical via −C, right = more sycophantic via +C); y = collateral "off-axis damage" (lower better). Black diamond = bare model at (0,0).
- Line = one method's dose sweep, smoothed (solid +C, dashed −C); dot = one dose; × = last dose still allowed by the judge (coherent, <5 pp false pushback, effect not reversed); ★ = plain prompt (declared in the caption, but I can't find any star on the chart).
- Grey = 5 random steering directions at the same doses: outer band 10–90%, inner 25–75%, grey line = median. Caption warns these are rank bands, not CIs.

*Chart 2 (discrimination.png)*
- Only the −C direction. x = extra pushback on nonsense questions (good); y = extra *false* pushback on sound-premise twins (bad). Red dotted line = 5 pp cutoff above which doses aren't scored.
- Lines/dots = method dose sweeps; stars = the two prompt baselines (plain prompt −C, engineered prompt −C); grey dots = random directions' mean per dose.

**2. Main message**
- Chart 1: steering shifts premise level in both directions, but the +C side reaches large shifts at damage levels largely inside what random directions achieve, while the −C side is cut off early (×) at modest shifts.
- Chart 2: on −C, every method rides roughly the same trade-off curve — more pushback on nonsense buys roughly proportional false pushback on sound twins — so within the 5 pp limit the gain is capped around ~1.0 and prompts blow far past the limit.

**3. Best method**
- +C: VJP-resid (× at ~1.9 shift, ~0.44 damage) and VJP-value (~1.6, ~0.38) sit at or just beyond the right edge of the grey band; the rest (chars, linear_act, mean difference) are inside it. So "best" is only marginally better than random perturbation. prompt × gain +C is clearly worst (damage ~1.2).
- −C: all × marks cluster at −0.4 to −0.9 with ~0.2 damage; VJP-resid and eng. prompt × gain reach farthest (~−0.8/−0.9). Grey band barely extends left of 0, so there's little random reference there.
- vs prompt baselines: in chart 2, prompt −C (★ ~1.7 gain, ~25 pp false pushback) and eng. prompt −C (~1.9, ~17 pp) get bigger raw shifts but fail the 5 pp limit badly; every steering method crosses 5 pp at ~0.9–1.05 gain. Eng. prompt × gain has the gentlest slope at high dose (~11 pp at x≈1.4 vs 15–24 for others) but crosses 5 pp at about the same place.

**4. −C vs +C**
- +C (sycophantic) is easy: large shifts, moderate damage — but random directions do nearly as well, so it may just be generic degradation rather than targeted steering.
- −C (skeptical) is hard to do *cleanly*: low off-axis damage in chart 1, but chart 2 shows the skepticism is mostly contrarianism — false pushback rises almost as fast as real pushback, so legitimate −C shifts are capped near ~1 (random directions give none at all).

**5. Confusions / possible misreads**
- Caption says ★ = plain prompt, but chart 1 shows no star; prompt baselines appear only in chart 2.
- Grey band is described as "both signs" yet appears only on the +C side (x≈−0.3 to 1.8), so −C methods have no random reference — easy to misread as "−C beats random".
- "eng. prompt × gain +C" is labeled at x≈−0.4 (left side), i.e. +C moved the model *skeptical*; sign flipped or mislabeled.
- Chart 1 lines are "smoothed over neighbouring doses", and VJP-resid +C has a loop at x≈2 — dose ordering is unreadable; also linear_act +C has a faded duplicate × next to chars +C.
- Overlapping labels (VJP-value −C, linear_act −C, chars −C, mean difference −C all stacked at x≈−0.6) and chars/linear_act/mean-difference +C lines nearly coincide.
- Units are undefined: "premise level", "off-axis damage", "on-target weighted", "Jev"; no actual dose values anywhere.
- Random directions in chart 2 move *negative* on x (less pushback) — unclear whether that's meaningful or just noise around bare.
- The 5 pp limit in chart 2 is the thing that places the −C × marks in chart 1, but you have to infer that cross-chart link yourself.
