# Method source evidence

Module docstrings copied verbatim; descriptive claims may need checking against code. Compiled by PI/OpenAI.

## src/steering_lite/variants/angular_steering.py

Angular Steering: fixed-plane rotation in activation space.

Vu & Nguyen 2025 NeurIPS https://arxiv.org/abs/2510.26243

Construct a fixed orthonormal steering plane $(b_1, b_2)$ per layer:
$b_1$ is the normalized mean contrast direction and $b_2$ is the first PC of
normalized pairwise contrast directions, orthogonalized against $b_1$.
Set the projection's in-plane angle to $\theta$ while preserving its in-plane
norm (the paper's fixed-plane target-angle form):

$$h' = h - P h + \lVert P h \rVert (\cos\theta \, b_1 + \sin\theta \, b_2).$$

The orthogonal component is untouched and the in-plane norm is preserved, so
total residual norm is preserved. `cfg.coeff` is $\theta$ in radians.

## src/steering_lite/variants/chars.py

CHaRS: Concept Heterogeneity-aware Representation Steering.

Abdullaev et al. 2026 https://arxiv.org/abs/2603.02237

Generalises mean_diff to multimodal concepts: instead of one direction per
layer, learn K source clusters $a_i$, K target clusters $b_j$, an OT
coupling $P^*_{ij}$ between them via Sinkhorn, and gate the per-cluster
translations $v_{ij} = b_j - a_i$ by an RBF kernel on distance to source
centroids:

$$\hat v(x) = \sum_{ij} \frac{P^*_{ij} \, k(x, a_i)}{\sum_{pq} P^*_{pq} k(x, a_p)} (b_j - a_i)$$

with $k(x, a_i) = \exp(-\lVert x - a_i \rVert^2 / 2\sigma^2)$ and
$P^*$ from entropic OT: $P^* = \arg\min_P \langle P, C \rangle + \lambda H(P)$
with $C_{ij} = \lVert a_i - b_j \rVert^2$ and marginals $p, q$ proportional
to cluster sizes.

When K=1 this reduces exactly to mean_diff (one cluster each, P trivial,
kernel constant, single translation $b - a$).

Apply: $h \leftarrow h + \alpha \hat v(x)$ (additive form, Definition 3.1).

## src/steering_lite/variants/corda_pca.py

PCA in CorDA's context-oriented decomposition space.

CorDA (Yang et al. 2024) orients each Linear's weight decomposition with a
context covariance. For a Linear weight `W` and input-activation covariance
`Sigma_x`, CorDA decomposes:

$$W \Sigma_x = U S V^T$$

and reconstructs the original weight by applying the inverse covariance to the
right singular factor:

$$W = U S (\Sigma_x^{-1} V)^T$$

The reference implementation uses full-sequence covariance matrices and notes
large memory use for 7B-scale models. This steering variant uses the same
decomposition algebra on the last-token prompt activations used by the rest of
steering-lite, with Tikhonov damping:

$$\Sigma_\lambda = X^T X / n + \lambda I$$

Then it runs PCA on paired differences in the adapter hidden coordinate:

$$z = x (\Sigma_\lambda^{-1} V_r) \sqrt{S_r}$$

and applies a constant hidden-coordinate nudge:

$$y \leftarrow y + \alpha (v_z \sqrt{S_r}) U_r^T$$

Refs:
  - CorDA paper: https://arxiv.org/abs/2406.05223
  - Reference code: https://github.com/iboing/CorDA

## src/steering_lite/variants/cosine_gated.py

Cosine-gated mean-difference steering (CAST-inspired soft self-gate).

Same `v_L` as mean_diff, but the update is gated by how aligned the residual
already is with the steering direction. We use **|cos|** because we don't care
about sign (steering can flip a feature; what matters is overlap), and a **soft
gate** (relu shifted by tau) instead of CAST's binary condition.

$$h \leftarrow h + \alpha \cdot \hat{v}_L \cdot \max(0, |\cos(h, \hat{v}_L)| - \tau)$$

When `tau=0`, gate ∈ [0, 1] = |cos|, full proportional. When `tau=0.1`, only
fires for tokens with overlap > 0.1.

Refs:
    - Inspired by CAST / conditional activation steering. This is not IBM CAST:
        it uses the same vector for behavior and condition and a soft per-token gate.

## src/steering_lite/variants/directional_ablation.py

Mean-diff directional ablation (Arditi-inspired projection-out).

Project the steering direction *out of* the residual stream instead of (or in
addition to) adding to it. Unlike `mean_diff` which translates by $\alpha v$,
ablation removes the component of $h$ along $\hat v$:

$$h \leftarrow h - (h \cdot \hat v)\hat v + \alpha\hat v$$

When `coeff=0` this is pure ablation (refusal-direction style); when `coeff!=0`
this is ablation followed by a constant nudge (useful to ablate "old" behavior
and inject "new"). The two terms are mathematically distinct -- ablation is a
*projection* (idempotent), addition is a *translation*.

Norms shrink by $|h \cdot \hat v|$ which is informative -- a near-zero shrink
means the direction wasn't present in the first place, so the intervention is
a no-op. Compare to `mean_diff` which always pays a constant $\alpha\|\hat v\|$
per token regardless of whether the direction is present.

Refs / inspiration:
  - Arditi et al. 2024 "Refusal in language models is mediated by a single direction"
    https://arxiv.org/abs/2406.11717
  - andyrdt/refusal_direction https://github.com/andyrdt/refusal_direction

## src/steering_lite/variants/kv_cache_gram.py

Contrastive steering of the attention value cache in a Gram basis.

This method changes the persistent value-cache path rather than a block's
residual output. Fit uses actual cached values from positive and negative
prompts. For every selected layer and KV head, it streams prompt-normalized
second moments and class means:

    G = 1/2 E_pos[V^T V / T] + 1/2 E_neg[V^T V / T]
    B_r = top-r eigenvectors(G)
    c = normalize(B_r B_r^T (mean_pos[V] - mean_neg[V]))

The Gram eigenspace is the right-singular subspace of the concatenated cache in
exact arithmetic. It identifies high-energy value directions; the class
contrast selects the direction inside that subspace.

At inference, keys and attention weights at the edited layer are unchanged. New
values are changed before insertion into the cache, while values in a supplied
ordinary prefix cache are changed once when that cache is promoted:

    z_t = V_t c
    V'_t = V_t + coeff |z_t| c

Positive coefficients amplify positive projections and reduce negative ones;
negative coefficients do the reverse. Orthogonal value content is preserved.
No optimizer or backward pass is used.

Selected layers must use transformers full-attention DynamicLayer caches. Hybrid
models work when only their full-attention layers are selected. Promoting an
already-populated hybrid, static, sliding-window, or quantized cache is unsupported.

## src/steering_lite/variants/linear_act.py

Linear-AcT: coordinate-wise affine activation transport.

Rodriguez et al. 2025 ICLR, "Controlling Language and Diffusion Models by
Transporting Activations" https://openreview.net/forum?id=l2zFn6TIQi

Linear-AcT fits one univariate affine OT map per activation coordinate. For
source samples $a_i$ and target samples $b_i$, sort each coordinate, centre the
sorted values, and fit

$$T_j(x_j) = \omega_j x_j + \beta_j$$

with

$$\omega_j = \frac{\sum_i \tilde a_{(i),j} \tilde b_{(i),j}}{\sum_i \tilde a_{(i),j}^2}, \quad
\beta_j = m_{b,j} - \omega_j m_{a,j}.$$

Apply strength $\alpha$ by interpolation:

$$h \leftarrow (1-\alpha)h + \alpha T(h).$$

This is the paper's core coordinate-wise `Linear-AcT` map, not the multivariate
Gaussian/Bures OT map. It omits optional support masking and layerwise map
estimation from the full pipeline.

## src/steering_lite/variants/mean_diff.py

Mean-difference steering (CAA / ActAdd).

For each selected layer L, compute the mean difference between positive and
negative last-token hidden states:

$$v_L = \text{mean}(h^+_L) - \text{mean}(h^-_L), \quad \hat{v}_L = v_L / \|v_L\|$$

At runtime, add `coeff * sum_i v_i` to every token's residual at that block:

$$h \leftarrow h + \alpha \cdot \sum_i v_i$$

Linear method: stacked rows can be summed at apply time, equivalent to
applying each round sequentially.

Refs:
  - Panickssery 2023 (CAA) https://arxiv.org/abs/2312.06681
  - Turner 2023 (ActAdd) https://arxiv.org/abs/2308.10248
  - Jorgensen 2024 (Mean-Centring) https://arxiv.org/abs/2312.03813

## src/steering_lite/variants/pca.py

PCA steering (RepE/LAT-inspired, vgel pca_diff-like).

For each layer L, compute PCA on the **paired differences** `h^+ - h^-`. Take
the top principal component as the steering direction.

$$D_L = H^+_L - H^-_L \in \mathbb{R}^{n\times d}$$
$$U, S, V^T = \text{SVD}(D_L - \bar{D}_L)$$
$$\text{sign}_L = \text{sign}(\bar{D}_L \cdot V_{:,0})$$
$$v_L = V_{:,0} \cdot \text{sign}_L$$

Sign-fixed by aligning the sign-ambiguous top PC to the MEAN paired-difference
(the persona contrast hs), so +coeff always moves toward the positive pole. This
matches repeng's orient-to-positive-class rule (it projects the uncentered
hiddens and flips if pos-mean < neg-mean) and AntiPaSTO's sign(mean(diff_S)).
(Claude 2026-07-15) The prior sign rule voted on CENTERED projections, whose mean
is zero by construction, so it measured the variance cloud's skew rather than
concept polarity and flipped the steering direction at random -- see
AntiPaSTO_concepts/README.md:577-582. This is a lightweight control-vector
baseline, not the full Zou et al. LAT reader: it omits per-diff normalization,
label-based sign selection, and train-mean recentering for reading scores.

At runtime, add `coeff * v_L` to the residual.

Refs:
  - Zou et al. 2023 (Representation Engineering) https://arxiv.org/abs/2310.01405
  - vgel/repeng: https://github.com/vgel/repeng

## src/steering_lite/variants/query_steer.py

Query steering: mean difference of attention queries (wassname/superkv, renamed query-steering).

For each selected attention layer L, capture every head's query after `q_norm` and before RoPE at the last
real token of each prompt, and take the class difference:

$$q^*_L = \text{mean}(q^+_L) - \text{mean}(q^-_L) \in \mathbb{R}^{H \times d_{head}}, \quad \hat q^*_L = q^*_L / \|q^*_L\|_F$$

At runtime add it to every position's query:

$$q \leftarrow q + \alpha \cdot \hat q^*_L$$

Keys and values are unchanged, so a head can only change which tokens of the current context it reads
(and how sharply); it cannot write new content the way residual steering does.

Differs from the superkv repo, which adds q* at the last token only: here every position is steered
(steering-lite convention; the teacher-forced KL calibration needs every position steered).
Requires `self_attn.q_norm` (Qwen3 / Qwen3.5); on hybrid models select only full-attention layers.

Ref: https://github.com/wassname/superkv (README "Query steering")

## src/steering_lite/variants/random.py

Random-direction null steering (placebo).

A per-layer random unit direction, independent of the contrastive data, added
like mean_diff / CAA. Calibrated to the same iso-KL dose as every real method,
so its selectivity score is the floor from an arbitrary perturbation of equal KL
magnitude. If a real method doesn't clear this null, its "steering" is generic
disruption (any push of this size moves the axis), not a specific-axis move.

Seeded per (cfg.seed, layer) so directions are reproducible and independent
across layers; vary cfg.seed to draw the null distribution (a null is a
distribution, not one draw). (Claude, not wassname)

## src/steering_lite/variants/spherical.py

Ungated spherical steering core (slerp on residual hypersphere).

Treat the residual as a point on the (d-1)-sphere and rotate it toward a target
direction `v` by slerp fraction `coeff`. Slerp preserves norm except at the
antipodal degeneracy where the geodesic is not unique:

$$h_{\text{rot}} = \text{slerp}(\hat{h}, \hat{v}, \alpha) \cdot \|h\|$$

where slerp is

$$\text{slerp}(a, b, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} a + \frac{\sin(t\Omega)}{\sin\Omega} b, \quad \Omega = \arccos(a \cdot b)$$

This is the fixed-t, ungated core of Spherical Steering. It omits the paper's
vMF confidence gate (`kappa`, `alpha`, `beta`).

Refs:
    - Spherical Steering https://arxiv.org/abs/2602.08169
  - chili-lab/Spherical-Steering https://github.com/chili-lab/Spherical-Steering

## src/steering_lite/variants/sspace.py

Weight-SVD S-space steering, cosine-gated (AntiPaSTO arithmetic relaxation).

Standard activation steering adds a constant bias `h <- h + alpha * v`
regardless of input. Weight-SVD S-space steering operates in the SVD basis of
a Linear's *weight matrix*, so the perturbation is input-dependent: tokens
whose S-space representation aligns with the contrastive direction get
pushed; tokens that don't are left alone.

For a Linear `y = x W^T + b` with `W = U S V^T` truncated to top-r, the
whitened S-space coordinates are accessible from EITHER side:

    x V sqrt(S) = (y - b) U / sqrt(S) = x_S          # x_S == y_S

We use the *output* projection at both extract and apply time.

Multi-round:

  - SVD basis (U_r, sqrtS, b) is a property of the weight matrix and
    is `shared` across rounds. Each round's `dS_2`, `dS_3`, ... extracts in
    the SAME basis (since W is frozen), which is exactly what makes
    accumulation valid.
  - Stacked tensor `dS: [k, r]`. Each row is `alpha_i * dS_hat_i_unit`,
    so row magnitude carries that round's per-direction calibration. Apply
    normalizes on-the-fly.
  - Per-round gate: each direction keeps its own `|cos(xS, dS_i)|`. This is
    strictly more faithful than baking magnitudes into a single direction,
    because each contrast was calibrated under different conditions.

Apply (k stacked directions, all in the same basis):

    xS      = (y - b) @ U_r / sqrt(S)              # [b, s, r]   ONCE
    dS_hat  = dS / ||dS||_row                       # [k, r] unit
    alpha   = ||dS||_row                            # [k]   per-direction calib
    gate    = |cos(xS, dS_hat)|                     # [b, s, k]
    deltaS  = einsum(gate * alpha, dS_hat, "bsk,kr->bsr") * cfg.coeff
    y'      = y + (deltaS * sqrt(S)) @ U_r^T

Why cosine in S-space: in d=2560 the cosine of two random vectors
concentrates near 0; in r=64 the cosine is a meaningful per-token signal.

## src/steering_lite/variants/sspace_ablate.py

Weight-SVD ablation in S-space.

Companion to `sspace`: same extract path (SVD a Linear's weight, recover
whitened S-space coordinates from the output via `(y - b) @ U_r / sqrt(S)`,
compute contrastive mean-diff direction `d_S_hat`), but `apply` projects
`d_S_hat` *out of* `x_S` instead of nudging along it. The output
perturbation is then lifted back via `(delta_S * sqrt(S)) @ U_r^T`.

Math (per token, k=1):

    x_S      = (y - b) @ U_r / sqrt(S)         # whitened S-space coords
    proj     = (x_S . d_S_hat)                 # scalar component along contrastive dir
    delta_S  = -proj * d_S_hat (+ alpha * d_S_hat)   # ablation + optional nudge
    delta_y  = (delta_S * sqrt(S)) @ U_r^T     # lift back to out-space
    y'       = y + delta_y

Multi-round (k stacked):
  Naive sum of independent rank-1 ablations is *wrong* if directions overlap
  (the shared component would be subtracted multiple times). We orthonormalize
  the stack via QR, then project x_S onto that subspace and ablate the
  projection in one shot. For k=1 reduces exactly to the formula above.

Gating: none. The projection magnitude `(x_S . d_S_hat)` IS the input-dependent
strength, so a separate cosine gate would be redundant. Tokens with no
contrastive component get a near-zero ablation by construction.

Compare to:
- `directional_ablation.py`: ablation in full residual d-space. Removes the
  direction from h regardless of how aligned the model's actual computation
  is with that direction.
- `sspace.py` (cosine-gated additive): pushes along d_S_hat with a strength
  proportional to S-space alignment.

Together with `sspace`, this gives an additive vs subtractive comparison at
the same hook site / extract path.

## src/steering_lite/variants/sspace_damp_amp.py

Multiplicative damp/amp steering in S-space.

Companion to `sspace`: same extract path (full SVD, |dS|.topk(r) mode
selection, store U_r, sqrtS, dS_hat). At apply time, instead of
adding `alpha * gate * d_S_hat` we *multiply* per-mode singular values by
`exp(c * d_S_hat_i)`, so modes with positive contrastive sign get amplified
and modes with negative get damped.

Math (per token):

    For W = U Σ V^T,  y - b = sum_i (x v_i) σ_i u_i.
    y_orig^(mode i)  = (x v_i) σ_i u_i
    y_new^(mode i)   = (x v_i) σ_i exp(c d_S_hat_i) u_i        # multiplicative
    delta_y^(mode i) = (x v_i) σ_i (exp(c d_S_hat_i) - 1) u_i

    Identity: (x v_i) σ_i = (y - b) u_i  (row of (y-b) onto u_i)
    -> delta_y_i = ((y-b) u_i) * (exp(c d_S_hat_i) - 1) * u_i

In matrix form over r selected modes:

    proj_r   = (y - b) @ U_r                                    # [..., r]
    scale    = exp(c * d_S_hat).clamp_(±clamp) - 1              # [r]
    delta_y  = (proj_r * scale) @ U_r^T                         # [..., d_out]
    y'       = y + delta_y

Multi-round (k stacked):
  Composition of multiplicative effects = multiplication of scales = ADDITION
  of log-scales. So accumulating k rounds is the row-wise sum of stacked dS:

      log_scale = c * Σ_i (alpha_i * d_S_hat_i)         # [r]   (= c * stacked.sum(0))
      scale     = clamp_exp(log_scale) - 1
      delta_y   = (proj_r * scale) @ U_r^T

  For k=1 reduces to the single-round formula above.

Properties:
  - Monotone in c (larger |c| -> stronger effect).
  - S_eff = σ * exp(...) > 0 always; never sign-flips a mode.
  - At c=0 the steering is exactly identity (scale=0).
  - Non-selected modes contribute 0 (they are absent from U_r).
  - No cosine gate: the per-mode multiplier IS the gating signal (high-|dS_hat_i|
    modes get more amplification; low-|dS_hat_i| modes are nearly identity).

Compare to:
  - `sspace.py` (additive cosine-gated): adds `alpha * gate * d_S_hat`,
    sign-agnostic gate.
  - `sspace_ablate.py` (subtractive): projects d_S_hat *out* of x_S.

Hook target: `mlp.down_proj` (output-side); V is implicit via SVD identity.

## src/steering_lite/variants/sspace_pca.py

PCA in whitened weight-SVD S-space.

This is the clean ablation between residual PCA and cosine-gated S-space:

1. For each target Linear, decompose `W = U S V^T`.
2. Recover whitened S-space coordinates from the Linear output:

$$x_S = (y - b) U_r / \sqrt{S_r}$$

3. Run PCA on paired differences `x_S^+ - x_S^-`.
4. At apply time, add the PCA direction in S-space and map it back:

$$y \leftarrow y + \alpha (v_S \sqrt{S_r}) U_r^T$$

Unlike `sspace`, there is no cosine gate. The only question is whether the
PCA direction estimator is better after moving from residual space into the
weight-SVD coordinates.

## src/steering_lite/variants/super_sspace.py

Super-SVD S-space steering on the residual stream (cosine-gated, multi-vec).

Like sspace, but the basis is shared across many Linears. Where sspace SVDs
ONE weight matrix and steers in its column-space, super_sspace pools the
residual-side singular vectors of ALL writers and readers in the selected
blocks and SVDs the pool. The result is a global d_model -> r basis that
covers what the residual stream can hold from layer activity, not just one
Linear's slice.

Math (writers W with d_out=d_model, readers W with d_in=d_model):

    For non-square W = U Σ V^T:
      writer block:  B_l = U_l Σ_l        ∈ R^{d_model × k_l}
      reader block:  B_l = V_l Σ_l        ∈ R^{d_model × k_l}

    Direct: M = [B_1 | B_2 | ...] ∈ R^{d_model × Σ k_l}, then SVD M -> U_⋆.
    Cheaper: SVD via the Gram matrix:

      G = M M^T = Σ_l B_l B_l^T = Σ_l U_l Σ_l² U_l^T  (and V_l Σ_l² V_l^T)
                                    ∈ R^{d_model × d_model}

      eig(G) -> λ_⋆, U_⋆;  Σ_⋆ = sqrt(λ_⋆)

    G is d_model × d_model (e.g. 2560²) regardless of how many Linears we
    pool. Same U_⋆, Σ_⋆ as direct SVD of M.

Per-layer dS (residual-stream activations at hook layer L, n examples each):

    xS_pos = h_pos[L] @ U_⋆ / sqrt(Σ_⋆)
    xS_neg = h_neg[L] @ U_⋆ / sqrt(Σ_⋆)
    dS     = mean(xS_pos) - mean(xS_neg)
    select top-r modes by |dS| (task-specific, same as sspace).

Apply (block residual hook, like mean_diff):

    xS      = h @ U_⋆_r / sqrt(Σ_⋆_r)
    gate    = |cos(xS, dS_hat)|
    delta_S = α * gate * dS_hat
    delta_h = (delta_S * sqrt(Σ_⋆_r)) @ U_⋆_r^T
    h'      = h + delta_h

Compare to sspace: per-Linear basis vs global pooled basis. Hypothesis: the
pooled basis is more robust because tail directions of any single Linear are
noisy, but the *consensus* of writers/readers is cleaner.

Multi-round: basis is purely W-derived (Gram eigh of
writers/readers), so the shared-basis invariant holds across iterated rounds.
dS lives in `stacked` with leading [k, r] dim; apply mirrors sspace's einsum
pattern -- per-direction cosine gate, per-direction calibration via row norm.

Square Linears (e.g. Qwen3 o_proj / q_proj where d_in=d_out=d_model) are
ambiguous (writer or reader?) and are skipped by shape detection. Pass a
custom `fallback_regex` if your arch has unusual shapes.

## src/steering_lite/variants/svdkv.py

svdkv: steer by letting the query choose between two halves of the attention sink (+ optional mean-diff residual).

Most heads put much of their attention on the first token (the attention sink). svdkv hides the real sink and puts two
copies in the KV cache, one carrying +v* and one −v*, with keys a small step apart along u. A query shift C·u moves
attention between the halves, so C sets how much of ±v* the heads read. v* is the value mean diff (sycophantic − abrasive)
at the last token. svdkv_resid adds mean_diff's residual vector at the same C.

Per attention layer L (full-attention layers ≥ 1), per KV head g, all after RoPE:
    sink±   key k_first ± ε·u,   value v_first ± ν·v̂*,   logit bias −ln 2 each   (k_first, v_first: this row's real first
            token, read from the cache at runtime)
    real first token hidden from query positions that see ≥ 4 real tokens (earlier ones keep it, so the sink itself is not rewritten)
    q_t += C·u                                   u: least-variance direction of real keys and queries, ⟂ mean sink key
    head write ≈ w_sink · ν · tanh(ε·(q_t·u + C)·scale) · v̂*
C = 0 is near, not exactly, bare: the halves split on q_t·u + C, and u is only approximately ⟂ the model's queries.
Measured on Qwen3.5-0.8B, ν = 320, 8 dev-cohort chat prompts: KL(bare || C=0) 0.005 nats averaged over positions, 0.014 at
the last token (a calibrated dose C0 is 1 nat); max |Δ log-prob| 3.85 (a rare token). Qwen3-4B, ν = 12.7: 0.015 nats over
positions, last token mean 0.18, max 1.43 (one of 8 prompts is a full dose off bare at its last token).
Qwen3.5-4B, ν = 550 (calibrated): 0.007 nats over positions, last token mean 0.015, max 0.043; max |Δ log-prob| 9.2.
TODO(PI[claude]): centre the split, e.g. subtract each head's mean q·u; untested.
svdkv_resid also:  h_L += C · r_scale · r̂*_L on mean_diff's default layers (20-80% depth)

Constants are set at extraction from iso-KL doses (1 nat RMS KL, steering-lite calibrate_iso_kl):
    ν_g = nu_mult · C0(sink value alone) · ‖v̂*_g‖        nu_mult 3.2 (Qwen3-4B: best −C dose 12.7 / C0 4.0)
    r_scale = C0(mean_diff) / C0(svdkv)                    so each part contributes at its own calibrated strength

Evidence (Qwen3-4B, BS-bench v2, Jev), write-up https://github.com/wassname/query-steering/blob/concept-steer/outputs/results.md
(svdkv_resid was named qslotr_sum there, svdkv q_slot_big): full 100 questions, −C side score svdkv_resid +3.94 vs mean_diff
+1.71, paired 90% CI of the difference [+1.79, +2.67]; 3 dev seeds agree; +C ties (Qwen3-4B already accepts most premises).
svdkv alone ≈ mean_diff. The random-vector control there was for a sibling method (sinkr_sum: a fixed v* written into the
sink value + residual): with a random unit vector in place of v* its −C gain fell to mean_diff's level. No random control
was run for svdkv / svdkv_resid themselves.
Requires attention that goes through transformers' ALL_ATTENTION_FUNCTIONS (Qwen3, Qwen3.5 full-attention layers).
PI[claude] 2026-09-29.

## src/steering_lite/variants/topk_clusters.py

Top-k cluster steering.

Cosine-assignment k-means on the paired diffs. At runtime, pick the centroid
most aligned with the current residual (max cosine similarity) and add it.

$$\{c_1, ..., c_k\} = \text{cosine-kmeans}_k(H^+ - H^-)$$
$$h \leftarrow h + \alpha \cdot c_{j^*}, \quad j^* = \arg\max_j \cos(h, c_j)$$

Multi-round:

  Stacked tensor `C: [k_rounds, n_clusters, d]`. Each round runs its OWN
  argmax routing (one centroid pick per token per round) then sums deltas.
  This preserves per-round routing semantics -- different rounds may route
  the same token to different cluster sets, which is exactly the point of
  iterating: round 2 picks up modes round 1 missed under the perturbed
  residual.

## src/steering_lite/variants/vjp_cache.py

VJP-delta in cached value space, using the same target contrast and class difference.

Pullbacks are taken through the actual values returned by DynamicCache.update,
not through a projection-module proxy. Keys and recurrent states are not edited.
Source estimator: https://github.com/wassname/vjp-steering (efcd848).
Implementation: PI/OpenAI.

## src/steering_lite/variants/vjp_delta.py

VJP difference from vjp-steering efcd848 (the reference calls it vjp_delta).

c = mean(h_target_positive) - mean(h_target_negative)
v_layer = mean_positive(J.T @ c) - mean_negative(J.T @ c)

This follows the pinned estimator directly; direction signs are not reoriented by
an activation-cosine heuristic. Source: https://github.com/wassname/vjp-steering
Adapted to steering-lite registration by PI/OpenAI.
