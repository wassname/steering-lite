## Read-only visual review

**Verdict:** The near-bare dots and random-reference shading are visible. The standalone plot and browser screenshot are visually consistent in their main content, with qualifications below.

### Observations
- **Lower-dose region:** Both main views show multiple colored dots close to the bare diamond, especially around off-axis change ≈0.2–0.6. These visibly populate the approach toward bare rather than leaving only long, unmarked curves. Some dots overlap; a remaining line-only interval connects bare to the closest cluster around ≈0.2.
- **Random shading:** The nested gray bands, solid median, and dashed/dotted boundaries are clearly visible in both views. The standalone caption explicitly identifies the bands as random-direction first-crossing quantiles—not confidence intervals or density contours—and notes the absence of a control penalty for random.
- **PNG/browser agreement:** Selected methods, curve shapes, near-bare clusters, prompt stars, and terminal crosses agree qualitatively. Axis ticks, vertical extent, label placement, and rendering size differ; these are not pixel-identical presentations.
- **Interpolation disclosure:** The standalone caption says “lines interpolate between doses; dot = measured seed mean.” It does **not** claim every point along a segment was measured. The browser screenshot ends before the below-plot caption, so equivalent browser disclosure cannot be verified from this image.
- **Controls panel:** Many dots cluster near the bare control rate (roughly 3%). Farther right, several traces rise markedly. This panel makes the distinction between pushback gained and legitimate questions incorrectly called nonsense visible.

### Presentation caveats
- The browser introduction asks how far steering can go “before the answers break.” That is stronger than the standalone caveat, “Off-axis magnitude is not a coherence test.” Readers viewing only the browser screenshot could conflate off-axis change with demonstrated answer failure.
- Smooth connecting curves can visually suggest continuously established behavior, particularly where markers overlap or are widely spaced. The standalone caption appropriately limits that interpretation.
- Angular and spherical appear as inactive browser toggles, not plotted traces. These images cannot establish either angular’s lack of a common admissible dose or spherical −C’s newly passing doses. Inactivity should not be read as evidence that a method has no passing points.

### Limits
Images alone cannot verify marker provenance, seed averaging, completeness of measured doses, absence of filtering/thinning, or the precise interpolation algorithm. The corrected implementation description is compatible with what is shown, but is not independently proven by these screenshots. This is a presentation review, **not a scientific-validity assessment**.