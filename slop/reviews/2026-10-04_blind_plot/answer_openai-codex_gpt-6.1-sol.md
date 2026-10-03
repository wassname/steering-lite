[pi-web-access] Dynamic tool activation requires Pi 0.86.1 or newer; web tools remain eagerly available.
- **1. What they measure and symbols**
  - **First chart:** change in premise acceptance/rejection versus “off-axis damage.” Left means more skeptical; right means more sycophantic; higher on the page means less damage.
  - **Second chart:** extra rejection of nonsense versus extra, unwanted rejection of sound premises, for **−C**. Farther right and lower is better.
  - **Dots:** tested doses. **Lines:** dose sweeps; the first chart says these are smoothed. **×:** last dose passing the judge’s limits, or before the effect reverses past bare—not necessarily the best point. **Stars:** plain-prompt baselines. **Black diamond:** unmodified “bare” model.
  - **Grey:** random-direction comparison. First chart: median and percentile bands from five directions, **not confidence intervals**. Second chart: mean-per-dose dots, not a region.

- **2. Main messages**
  - **First:** steering can move the model substantially toward sycophancy, especially with VJP-resid, whereas acceptable skeptical movement is smaller.
  - **Second:** stronger skeptical steering eventually becomes indiscriminate pushback against sound premises.

- **3. Which looks best?**
  - **For +C, VJP-resid:** it reaches farthest right with relatively low damage, looking better than the grey region and much better than the ordinary prompt baseline.
  - **For useful −C, engineered prompt × gain looks strongest in the second chart:** it gets roughly one premise level of nonsense rejection while staying below the 5-percentage-point false-pushback limit.
  - That looks better than the random-direction dots and both plain-prompt stars for *selective* skepticism. The stars get more rejection, but also much more false pushback.
  - These are visual comparisons, not demonstrated statistical superiority.

- **4. Left versus right**
  - **+C has a larger usable range; −C is constrained by false rejection of sound questions.** Stronger skepticism is not automatically better discernment.

- **5. Confusions / possible misreadings**
  - The first chart’s vertical axis is inverted: “up” is better.
  - The second chart’s rightward movement means **more skepticism**, unlike the first chart.
  - Curves continue beyond the × marks or above the red limit; those extreme points should not be read as acceptable results.
  - “Off-axis damage,” weighting, and the judge’s exact limits are not fully defined.
  - First-chart stars are hard to distinguish amid overlapping markers and labels.
  - Grey shading could look like uncertainty, but represents only five random directions.
  - Dose values are absent, so matching operating points across charts is difficult.
