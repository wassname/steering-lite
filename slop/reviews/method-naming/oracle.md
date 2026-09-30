# Independent naming opinion (method families)

**Principle.** Names should encode *where* the intervention sits and *how* the direction is built, in a short, composable vocabulary that matches existing modules (`mean_diff`, `pca`, `query_steer`). Prefer `site_estimator` or `site_operator` over new acronyms. Keep published names (CHaRS, Linear-AcT, CorDA, CAA) as attribution prefixes, not as the whole identifier. Plot legends and `variants/` paths should stay under ~20 characters when possible.

**VJP: estimator vs site.** Distinguish **site**, not the pullback. Both modules use the same class-difference-of-VJPs estimator (`c` on target hiddens; `mean_pos(Jᵀc) − mean_neg(Jᵀc)`). What differs is the tensor that is edited: residual activations vs cached **values**. `vjp_delta` is the reference name and is already short; keep it for residual apply. `vjp_cache` is misleading (sounds like “cache the VJP”). Recommend `vjp_delta` / `vjp_value` (or `vjp_resid` / `vjp_value` if you want a matched pair). Do **not** encode “delta” only on one of them; both are delta-of-pullbacks. `vjp-delta_value` is too long for legends; `vjp_v` is too cryptic next to `svdkv`. Brevity fails when `vjp` is used alone: it does not say residual vs value.

**CorDA-derived.** `corda_pca` is honest if you keep the paper name as *algebra*, not as the training method. Docstring: last-token Tikhonov context second moments, then PCA of paired diffs in adapter coordinates, constant nudge. Recommend `corda_pca` unchanged, or `corda_adapt_pca` only if reviewers confuse it with original CorDA training. Do not drop “CorDA”; that would hide attribution.

**S-space family.** Keep `sspace` as the stem (weight-SVD whitened coords of **one** Linear). Current names already compose well:

| current | recommend | why |
|---|---|---|
| `sspace` | keep | cosine-gated additive in one-W S-space |
| `sspace_ablate` | keep | subtractive companion |
| `sspace_damp_amp` | `sspace_scale` | shorter; still multiplicative on modes |
| `sspace_pca` | keep | PCA estimator in same coords, no gate |
| `super_sspace` | `sspace_pool` | “super” is informal; pooling residual-side writer/reader factors is the actual distinction |

`sspace_pool` still needs a one-line legend (“pooled residual SVD basis”). Do not rename the family to AntiPaSTO; the docstring calls it an arithmetic relaxation, not the paper method.

**Attention-sink query steering.** `svdkv` is the most misleading name in the set. SVD is not the defining mechanism; the method splits the **sink** into two synthetic KV slots with opposite value contrast and shifts **queries** along a low-variance key/query direction. Recommend `sink_split` (or `q_sink_split`). `svdkv_resid` → `sink_split_resid` (additive mean_diff on residual at the same coefficient). `query_steer` is already good (Q-mean-diff, every position); do not fold it into `svdkv`.

**Other misleading leftovers (from supplied docstrings only).**

- `kv_cache_gram`: accurate (Gram of **values**, edit V). Optional `value_gram` if you want parallelism with `vjp_value`.
- `cosine_gated`: CAST-inspired but **not** IBM CAST; keep the descriptive name, not `cast`.
- `directional_ablation`: Arditi-inspired residual project-out; `resid_ablate` would pair with `sspace_ablate`.
- `pca`: not full LAT/RepE; `pca_diff` would match the paired-difference estimator and vgel’s usage.
- `spherical` / `angular_steering`: paper cores with documented omissions (no vMF gate; fixed-plane θ). Keep paper stems.
- `chars`, `linear_act`, `mean_diff`, `random`, `topk_clusters`: already semantic.

**Missing information.** No independent code check beyond parent facts and these docstrings. Hook sites for `sspace*` (e.g. `down_proj`) and hybrid-cache constraints should not enter names.

**Minimal worthwhile set:** `vjp_cache` → `vjp_value`; `super_sspace` → `sspace_pool`; `svdkv`/`svdkv_resid` → `sink_split`/`sink_split_resid`; optionally `sspace_damp_amp` → `sspace_scale`. Leave the rest.

— pi-quick-oracles (second opinion; independent of parent replacements)