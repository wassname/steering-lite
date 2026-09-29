## 1. Reading each plot

All three plots measure **change from the unsteered model**, not absolute answer quality. Horizontally, left means less premise acceptance (−C, labelled “abrasive”); right means more acceptance (+C, “sycophantic”). Vertically, damage increases **downward**: better means farther toward the intended side while remaining near the top. Neither horizontal direction is inherently “better”; that depends on the requested steer.

- **Qwen3.5-4B:** VJP-cache looks best on the left, achieving substantial movement with relatively little damage. On the right, VJP-delta looks slightly better than VJP-cache at their highlighted rings. VJP-cache wins overall under the README’s weaker-direction score.
- **Qwen3.5-27B:** Mean difference looks best on the left. VJP-cache appears best on the right by movement minus damage; mean difference reaches farther but incurs substantially more damage. Mean difference wins overall because its weaker direction is stronger.
- **OLMo-2-32B:** Mean difference looks best among activation methods on both sides, with chars next. VJP trajectories remain close to the vertical axis, especially leftward. Mean difference wins overall. Engineered prompting moves much farther left, but the README excludes prompting from scored comparisons.

These side-specific rankings are visual estimates, applying the README’s subtraction rule; the plots alone do not explain the overall aggregation. I also cannot tell from the images exactly how Jev’s units are defined.

## 2. Comparison with the README

The 4B winner does not transfer: mean_diff overtakes VJP-cache on both larger models. Qwen27B shows much more rightward than leftward movement; OLMo shows little clean movement, particularly for VJP methods. Both README tables support the overall ranking reversal and unchanged rankings after room correction. This is not evidence that size alone causes the changes, especially across model families.

One sentence in `README.md`, “Larger models,” overstates the evidence:

> “On OLMo they do not steer at all, while mean_diff and chars still do.”

The OLMo plot shows small, nonzero VJP movements, and the table gives vjp_delta **+0.03** on ÷ room. “Little useful steering” would fit better. The claim that bare Qwen27B rejects 69 premises cannot be independently checked from these plots or tables.

## 3. Readability issues, most important first

1. **Different axis scales:** Qwen27B spans much greater horizontal change; OLMo magnifies damage vertically. Similar-looking slopes are not comparable.
2. **Endpoint labels versus score rings:** labels often point to × endpoints, not rings, making the best operating point easy to misidentify.
3. **Crowding:** OLMo’s central rings/curves and Qwen27B’s rightmost VJP markers overlap.
4. **Unexplained gray region:** “null zone” lacks a stated statistical definition.
5. **Changing method selection:** 4B omits mean_diff, preventing direct visual tracking of the later winner.