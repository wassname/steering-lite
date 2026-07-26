# `vjp_delta` port UAT

Recorded by CODEX on 2026-07-25.

## Claim

SteeringLike's `feat/corda-space-steering` branch now contains wassname's
`vjp_delta` steering adaptation of Anthropic's public
[Jacobian Lens](https://github.com/anthropics/jacobian-lens). It can extract,
apply, calibrate, accumulate, save, and reload the direction through the
normal `Vector` interface.

Public reference implementation:
[`wassname/j-steer-dev/src/jsteer/variants/vjp.py`](https://github.com/wassname/j-steer-dev/blob/main/src/jsteer/variants/vjp.py).

Parity-test source snapshot: j-steer-dev `89d88e464eb9fa04a0fd04772bed78b26828b9e7`.

Target parent: steering-lite `dfb1624e4cb64045951160cb6538a5d2f7cc157a`.

The method begins with the usual target-layer contrastive vector and treats it
as a cotangent. For each selected earlier layer, a vector-Jacobian product
pulls that contrast back through the model without materializing the full
Jacobian. Subtracting the positive- and negative-class mean pullbacks gives the
source-layer steering direction.

The branch also exports the newer prompt-free `readout_words` diagnostic from
j-steer-dev. It maps each readable residual direction through the model's final
norm and unembedding, reporting associated +v and -v word tokens. SteeringLike
additionally reconstructs Super S-space's applied residual direction before
the readout instead of decoding its internal S coordinates.

## Evidence

| Check | Observable result | Verdict |
| --- | --- | --- |
| Source parity on cached two-layer random Llama | source-layer cosine `1.000000119`; maximum absolute element difference `0.000e+00`; both norms `1.0` | pass |
| Narrow functional smoke | `2 passed, 33 deselected in 13.79s` | pass |
| Full SteeringLike smoke | `35 passed in 85.77s` | pass |
| j-steer five-layer random-Qwen seam smoke | extraction split-half cosine `+0.832`; delivery resolved to `add`; run completed in 13 seconds | pass |
| Lint excluding the repository's known jaxtyping-string rules F722/F821 | `All checks passed!` | pass |
| j-steer registration seam | `REGISTRY["vjp_delta"].__module__ == "steering_lite.variants.vjp_delta"` | pass |
| Vector word readout | direct residual and reconstructed Super S-space tests recover the expected +v and -v words; narrow VJP/mean/SuperSSpace suite: `8 passed, 29 deselected in 6.13s` | pass |

The parity check loaded the same random model once, extracted the library
vector through `steering_lite.train`, extracted the source vector through
`jsteer.variants.vjp.extract_vjp_variants`, and compared layer 0 directly.
Both used target layer 1, `skip_first=0`, batch size 2, and the same eight
contrastive prompts.

The full smoke is `just smoke`. For every registered method it runs
extraction, finite calibration, a nonzero logit-effect assertion, and a
save/load logit round trip. Its multi-vector tests also include `vjp_delta`.

## Remaining interpretation limit

The smoke model has random weights. It proves the computation and storage
path, not that a persona axis is semantically precise or that steering changes
an organism's loyalty behavior. Those remain separate experiment gates in the
apart-secret spec.
