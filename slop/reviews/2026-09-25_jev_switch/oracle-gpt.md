## 1. Is Jev adequate?

**Provisionally, for this benchmark—not as independent ground truth. Confidence: high.** The brief reports “rank Spearman DeepSeek vs Jev 0.97” and strong answer-level agreement. That supports broad ranking stability. Determinism establishes repeatability, not validity; both judges can share biases, and the brief does not establish whether those agreement statistics cover the final nine-level rubric.

**The strongest risk is construct validity. Confidence: high.** The rubric change plausibly fixes a real floor effect, and penalizing content-free agreement follows “Rate vagueness as severe damage.” Nevertheless, detailed correction, verbosity, and rudeness may remain entangled: the blind labels include 33% “verbose” for vjp_cache’s negative side and 23% “rude” for chars. Those labels raise concerns; they do not prove mismeasurement.

**Cheap check:** blindly hand-label roughly 50 existing bare/steered pairs, stratified across methods, signs, initially rejecting questions, and doses around the damage cutoff. Separately assess specific correction, premise acceptance, vagueness, and style. Include concise and verbose paraphrases expressing identical judgments. Agreement without method-dependent residual errors would weaken this concern. Recompute rankings under reasonable penalty weights, damage caps, and alternative level spacing: ordinal categories do not automatically justify equal numerical intervals.

**“Beats random” needs a direct comparison. Confidence: high.** Bootstrap the method-minus-random difference using shared question resamples and matched experimental seeds where applicable. Individual intervals are not contrast intervals. Selecting doses and methods on the evaluation questions can inflate apparent performance; redoing selection inside bootstrap draws does not demonstrate held-out generalization. Cross-fit dose selection and evaluation using existing outputs.

## 2. Larger-model design

**A useful transfer experiment, but too narrowly winner-focused. Confidence: medium-high.** The brief says “Every method is limited by its −C side.” Thus the scalar ranking largely measures improved correction, not balanced evidence about both steering directions. The larger model may already reject more premises, changing available headroom.

For approximately the same walk count, use:

- vjp_cache, chars, vjp_delta, and **mean_diff**, three seeds each;
- eight random seeds;
- both prompt styles in both directions: four conditions.

That is 24 conditions; verify that prompt conditions really cost comparable “walks.” Drop linear_act for the simple mean_diff reference—not because linear_act is inferior, but because a simple baseline makes transfer easier to interpret. Reducing random seeds modestly sacrifices precision; inspect their existing spread first.

Report both directional scores, absolute bare/steered acceptance, damage distributions, and results split by whether bare already rejects. Lock rubric and analysis before running. First time one learned walk and one random walk on 27B; the brief’s uncertain scaling is not a reliable spending cap. **Confidence: high.**

## 3. Other risks

**Scope and uncertainty need explicit limits. Confidence: high.** All-false-premise questions cannot distinguish useful skepticism from indiscriminate contradiction; add a small matched valid-premise control. Three extraction seeds weakly characterize seed variability. Question bootstrapping omits rubric uncertainty, and mean damage can hide severe failures concentrated on a subset. None of these establishes a bug; raw outputs, per-seed results, and implementation checks could reduce the concerns.