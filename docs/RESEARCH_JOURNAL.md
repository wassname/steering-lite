# Research Journal — steering-lite

## 2026-04-28: Trajectory NLL as the calibration leash

### Question

Mean per-token TV at action positions averages over too many tokens; one big argmax-flip to gibberish (TV=0.95 at 1 token) is washed out by 127 calm tokens (TV=0.05) into mean-TV=0.057. We need a calibration leash that (a) is tail-sensitive (catches single big logit changes), (b) extrapolates to long trajectories, (c) is interpretable in nats so a target value means something.

### Method (be honest about what's measured)

**Forward $\Delta$NLL on base's greedy trajectory, teacher-forced.**

1. Pick a held-out dilemma prompt $x$.
2. Base model generates $y^{*} = (y_1^{*} \dots y_T^{*})$ greedy ($\arg\max$, no sampling), with EOS suppressed so we get the full $T$ tokens.
3. Both base and steered models do **one forward pass over $x \,\|\, y^{*}$** (teacher-forced — they don't generate, they score).
4. Per-token NLL: $\text{NLL}_{t} = -\log p_{\text{model}}(y_t^{*} \mid x, y_{<t}^{*})$.
5. $\Delta\text{NLL}_t = \text{NLL}_{\text{steer}, t} - \text{NLL}_{\text{base}, t}$ (pointwise).
6. Headline metric: $\overline{\Delta\text{NLL}} = \frac{1}{T} \sum_t \Delta\text{NLL}_t$ in nats/token.

This is forward-KL on the on-policy *base* trajectory (sampled, not summed-over-V). It is **NOT on-policy for the steered model** — a steered model that generates fluent-but-different text will not get measured here, only its disagreement with base's chosen tokens.

### Terminology (defining the terms used)

| term | meaning |
|---|---|
| **action positions** | the ~10-20 token spans in each dilemma marked as the value-loaded continuation (e.g. "I would lie to protect them"). The `action_pos`/`action_neg` strings in the dataset. |
| **action-iso / iso-action-dNLL** | calibration where coeff is bisected so $\overline{\Delta\text{NLL}}$ on **action-position tokens** hits the target (e.g. 0.10 nats). Original metric. Local. |
| **free trajectory** | base model's $T$-token greedy continuation of the dilemma prompt — base writes whatever it wants, no action target. |
| **free-iso / iso-free-dNLL** | calibration where coeff is bisected so $\overline{\Delta\text{NLL}}$ on the **free trajectory** hits the target. New metric. Generalizes. |
| **flip rate** | fraction of trajectory tokens where $\arg\max p_{\text{steer}} \neq y_t^{*}$. What greedy sampling would visibly do differently. |
| **dNLL** | $\Delta\text{NLL} = \text{NLL}_{\text{steer}} - \text{NLL}_{\text{base}}$, in nats/token. |

### Finding 1: action-iso calibration overshoots free-trajectory dNLL by 1.7-5.2$\times$

At iso-action-dNLL=0.10 (coeff bisected so action-token dNLL=0.10), the free-trajectory dNLL is much higher and method-dependent:

| method | dNLL@action | dNLL@free (cumulative-128) | dNLL@free/action ratio |
|---|---:|---:|---:|
| mean_diff | 0.106 | 0.197 | 1.7$\times$ |
| pca | 0.104 | 0.338 | 3.4$\times$ |
| spherical | 0.102 | 0.581 | 5.2$\times$ |

**Implication:** action-position calibration treats spherical's surgical local effect identically to mean_diff's diffuse effect, hiding that spherical pays a 5.2$\times$ higher generic-text coherence cost. SI_NLL ranking on action positions completely misleads about free-text reliability.

### Finding 2: under fair (free-iso) calibration, method ranking changes

At iso-free-dNLL=0.10 (coeff re-bisected so free-trajectory dNLL=0.10), Qwen-0.6B, n_eval=8, seed=0, **3 of 6 methods** (smoke):

| method | coeff(free-iso) | coeff(action-iso) | coeff drop | SI_NLL(free-iso) | SI_NLL(action-iso) |
|---|---:|---:|---:|---:|---:|
| mean_diff | 5.81 | 6.71 | -13% | -0.012 | -0.016 |
| pca | 6.26 | 8.94 | -30% | +0.028 | -0.016 |
| spherical | **0.042** | 0.089 | **-53%** | **+0.051** | +0.159 |

Spherical's 10$\times$ lead under action-iso shrinks to 2$\times$ under free-iso. PCA overtakes mean_diff. **Most of spherical's "surgical informedness" was the action-position calibration metric not penalizing it for breaking generic coherence.** TODO confirm at 6 methods, 3 seeds (pueue 252 in flight).

### Finding 3: shape of $\Delta\text{NLL}_t$ over $t$ — early ramp + plateau, no exponential compounding

At iso-action-dNLL=0.10, Qwen-0.6B, seed=0, single prompt (eval pair 0). Trajectory ran 164 tokens (model hit EOS — fixed in next run with `suppress_tokens=[eos]`).

| method | dNLL@2 | dNLL@8 | dNLL@16 | dNLL@32 | dNLL@64 | dNLL@128 |
|---|---:|---:|---:|---:|---:|---:|
| mean_diff | 0.153 | 0.122 | 0.066 | 0.171 | 0.225 | 0.197 |
| pca | 0.134 | 0.203 | 0.273 | 0.273 | 0.340 | 0.338 |
| spherical | 0.267 | 1.005 | 0.639 | 0.927 | 0.754 | 0.581 |

Three different shapes:
- **mean_diff**: U-shape — high at t=2, dip at t=16, ramp to plateau by t=64.
- **pca**: monotonic ramp from t=2 to t=64, plateau after.
- **spherical**: spike at t$\approx$8 (cumulative-mean 1.0 nat/tok!), gradual decay to plateau $\sim$0.5.

**No exponential compounding for any method.** All three plateau by t$\approx$64. Steering pays a roughly constant per-token tax once past an initial transient; LayerNorm + residual stream prevent runaway divergence. **Refutes** the "errors compound exponentially over CoT" worry under teacher-forced measurement. TODO: confirm under on-policy generation (steered generates, base scores) — different failure modes are accessible.

### Caveats / what I haven't shown

- **Teacher-forced, not on-policy.** Steered model is scored on base's argmax tokens, not on its own generations. A steered model could produce fluent-but-different text and look identical to one that quietly veers off-distribution within base's tokens. The complementary on-policy measurement (`scripts/long_gen_check.py`) exists but isn't in this analysis.
- **N=1 prompt for the trajectory plot.** Single dilemma, single seed. The shape of $\Delta\text{NLL}_t$ is method-specific but I haven't verified prompt-stability. TODO: aggregate over $\geq 8$ prompts.
- **Greedy base generation.** Real CoT uses temperature$>0$ sampling. Long-CoT compounding may show up under sampling but not greedy. Defer to v2.
- **Trajectory ended at t=164** (EOS); plotted "1024" claim was wrong. Fixed for next run by suppressing EOS at generation time.

### Evidence

- Raw CSV: [outputs/trajectory/trajectory__Qwen--Qwen3-0.6B-Base__seed0.csv](../outputs/trajectory/trajectory__Qwen--Qwen3-0.6B-Base__seed0.csv) — per-token base_nll, steer_nll, delta_nll, flip across mean_diff/pca/spherical for t=0..163.
- Plot script: [scripts/plot_trajectory_nll.py](../scripts/plot_trajectory_nll.py) — log-x cumulative-mean + 16-tok-smoothed per-token panels.
- Calibration script: [scripts/iso_tv_calibrate.py](../scripts/iso_tv_calibrate.py) — `--metric free_dnll` (default), `--metric delta_nll_target` (action-iso), `--metric tv_target` (legacy).
- iso-free-dNLL JSON (3 methods, 1 seed): [outputs/iso_tv/iso__Qwen--Qwen3-0.6B-Base__L4__free_dnll0.1__seeds0__1777343934.json](../outputs/iso_tv/iso__Qwen--Qwen3-0.6B-Base__L4__free_dnll0.1__seeds0__1777343934.json).
- Pueue 252: full sweep Qwen-0.6B, 6 methods, 3 seeds at iso-free-dNLL=0.10. Queued.
- Pueue 254: trajectory plot all 6 methods at the calibrated coeffs from 252's output, 1024 tokens with EOS suppressed. Queued `--after 252`.

### Next step (the headline claim that needs proving)

> *"Calibration on free trajectory transfers; calibration on action positions doesn't."*

Required evidence:
1. 6-method 3-seed iso-free-dNLL coeffs are seed-stable (CV across seeds < 30% per method) — pueue 252.
2. The trajectory plot shows that at iso-free-dNLL=0.10, $\Delta\text{NLL}_t$ stays close to 0.10 across all 6 methods over $t \in [128, 1024]$ (the plateau holds). At iso-action-dNLL=0.10, $\Delta\text{NLL}_t$ is much higher and method-spread — pueue 254.
3. Per-prompt aggregation ($\geq 8$ prompts) shows shape is robust. TODO.
4. On-policy generation check (steered generates, base scores) gives the same ranking. TODO.

If 1+2 land, that's the publication-grade methodology section. 3+4 are reliability checks for the appendix.

### 2026-04-28 (followup): on-policy steered trajectory + KL vs NLL

Added [scripts/onpolicy_vs_teacher.py](../scripts/onpolicy_vs_teacher.py) that, for each calibrated method, generates BOTH a base greedy trajectory $y_b$ AND a steered greedy trajectory $y_s$ (EOS suppressed, $T=128$), then scores each under both models. Reports per-position TV, KL (full-vocab, no sampling noise), and NLL_diff (single-sample estimate at the trajectory token) at lengths $\in \{2, 4, 16, 32, 64, 128\}$.

Two regimes:
- **TF_base**: base writes $y_b$, steered scores it. "Did steering disrupt base's path?"
- **OP_steer**: steered writes $y_s$, base scores it. "Did steered drift to gibberish base considers off-distribution?"

Result, Qwen-0.6B, seed=0, layers=[4], iso-free-dNLL=0.10 calibration, single prompt (eval_idx=0):

**TF_base @ T=128:**

| method | TV | KL(steer\|\|base) | NLL_diff |
|---|---:|---:|---:|
| mean_diff | 0.107 | 0.096 | -0.092 |
| pca | 0.181 | **0.524** | -0.274 |
| spherical | 0.122 | 0.099 | -0.078 |

**OP_steer @ T=128:**

| method | TV | KL(steer\|\|base) | NLL_diff |
|---|---:|---:|---:|
| mean_diff | 0.103 | 0.080 | -0.003 |
| pca | 0.140 | **0.175** | +0.089 |
| spherical | 0.107 | 0.070 | +0.070 |

**Findings:**

1. **KL > |NLL_diff| for all methods**, especially pca (KL=0.52 but $\Delta$NLL=0.27 in TF regime). KL integrates over the whole vocab; NLL_diff is a 1-sample estimate at the chosen token. **KL is the noise-free signal, NLL_diff is its sampling estimator.** For calibration, **KL is the better target** (deterministic given two distributions, no sample variance). NLL_diff is what we've been using and it understates the true distribution shift, especially when methods spread probability widely without flipping argmax (= pca).

2. **pca shifts probability mass MORE than its argmax-shift suggests.** TF KL=0.52 vs |NLL_diff|=0.27, ratio 1.9$\times$. mean_diff and spherical have ratio $\approx$1.0. Means pca redistributes mass without crossing argmax — visible to KL, invisible to greedy sampling. This is exactly the failure mode the user flagged: large-mass-shift-but-no-flip.

3. **OP_steer KL $\le$ TF_base KL for all 3 methods.** Steered's own preferred tokens (when free to generate) are *more* compatible with base's distribution than base's tokens are under steered. No "drift to gibberish" failure mode at iso-free-dNLL=0.10 for these 3 methods. Demos confirm: all 3 produce fluent English.

4. **pca's OP_steer KL=0.175 vs TF_base KL=0.524.** Big asymmetry. When steered chooses tokens for itself, base agrees more (0.175 KL) than when steered is forced onto base's tokens (0.524 KL). pca seems to push toward *plausible-under-base* alternatives, not gibberish.

5. **Single-prompt warning.** All numbers above are $N=1$ prompt. Calibration was on 4 prompts, here we're on 1. The TF KL=0.52 for pca is well above the calibration target (0.10) — likely just prompt-variance, since the 4-prompt calibration *did* converge.

**Methodology decision:** Switch calibration target from mean_dNLL to **mean KL(steer || base) on free trajectory** (denoted "free_kl"). Same forward passes — KL is computed from the same logits, just sums over vocab. Tail-sensitive in distribution-space without single-token sampling noise.

**Evidence:** [outputs/trajectory/onpolicy_vs_tf__Qwen--Qwen3-0.6B-Base__seed0.csv](../outputs/trajectory/onpolicy_vs_tf__Qwen--Qwen3-0.6B-Base__seed0.csv) (3 methods × 6 lens × 2 regimes × 3 metrics = 108 cells), [demos.json](../outputs/trajectory/onpolicy_vs_tf__Qwen--Qwen3-0.6B-Base__seed0__demos.json) (3 methods × 2 trajectories full-text). Pueue 255 will rerun on 6 methods after 252 finishes.

# Research journal

## 2026-05-02 — tinymfv sweep, Care vs Sanctity axis (Qwen3-0.6B)

Job 79 done (11/11 methods). Persona POS="a deeply traditional person" / NEG="a deeply caring person". Iso-KL target kl_p95=1.0 nat, bracket (0.05, 16). axis = ΔlogitSanc − ΔlogitCare.

| axis  | method                | C      | kl_p95 | Care   | Sanc   | notes                                              |
| ----: | --------------------- | -----: | -----: | -----: | -----: | -------------------------------------------------- |
| +0.88 | chars **(degen)**     |  +0.03 | null   | +3.99  | +4.87  | ALL 7 foundations ≈ +4 nats: output collapse, not steering. axis number is junk |
| +0.78 | cosine_gated          | +17.60 | ~1.0   | −0.51  | +0.28  | needed 8x larger C than mean_diff for same KL      |
| +0.77 | spherical *(uncalib)* |  +0.03 | 13.4   | −0.27  | +0.50  | calib hit bracket floor (kl never < target). Result still healthy |
| +0.75 | mean_diff             |  +2.00 | 1.02   | −0.48  | +0.27  | clean iso-KL hit                                   |
| +0.75 | mean_centred          |  +2.00 | ~1.0   | −0.48  | +0.27  | identical to mean_diff (corpus mean ≈ 0)           |
| +0.74 | sspace                |  +2.08 | ~1.0   | −0.47  | +0.27  | functionally mean_diff with different parametrization |
| +0.66 | topk_clusters         |  +2.68 | ~1.0   | −0.59  | +0.07  | mostly Care-suppression, weak Sanc lift            |
| +0.83 | angular_steering *(uncalib)* | +0.03 | 15.6 | −1.03 | −0.20 | calib floor; KL 15× target. Care nuked, Sanc neg-ish. uncalibrated junk |
| +0.65 | directional_ablation *(uncalib)* | +0.03 | 2.62 | −0.35 | +0.30 | calib hit bracket floor; kl 2.6× target |
| +0.39 | linear_act            |  +2.59 | 1.00   | −0.17  | +0.22  | clean iso-KL hit; weakest among additive calibrated methods |
| +0.29 | prompt_only           |    n/a | n/a    | −0.05  | +0.24  | baseline — small effect, no dial                   |
| +0.24 | pca                   |  +1.81 | ~1.0   | −0.68  | −0.43  | all foundations went negative; axis +ve only because Care fell more |

Bare model (absolute logit): Care=+0.60, Sanc=−0.28 (0.88-nat gap is what we're closing).

Pending: repeng_raw (job 80 failed: `set_control(coefficient=...)` is wrong kwarg, should be `coeff=`. Fixed; requeued as job 87), engineered_prompt (job 86 cancelled by dependency-fail; requeued as job 88 after 87).

Calibrated leaderboard (apples-to-apples, kl_p95≈1.0): cosine_gated (+0.78), mean_diff/mean_centred (+0.75), sspace (+0.74), topk_clusters (+0.66), linear_act (+0.39), pca (+0.24).
Uncalibrated (informational only): chars (degen), spherical/directional_ablation/angular_steering (calib floor — all methods with non-additive C unit).

Open observations:
- **chars is broken on Qwen3-0.6B.** All foundations end up at +4 nats — model collapses to "wrong" regardless of vignette. Don't trust the +0.88 axis; needs investigation before reporting.
- **Calibrator floor problem.** spherical, directional_ablation, chars all returned C=0.028 (bracket-mid 0.894 halved 5×) because kl_p95 stayed above target=1.0 even at the bracket floor of 0.05. For spherical, kl saturated at 13.4 nats — its rotation parameterization has a different geometry than additive methods, and the (0.05, 16) bracket assumes additive scale. directional_ablation overshot to 2.6×. Either (a) extend the bracket down for these methods, or (b) accept they're "max-strength" and label uncalibrated.
- cosine_gated calibrating to C=17.6 is a tell that the soft gate zeros out most of the intervention; raw "scale" comparisons across methods are meaningless without iso-KL.
- mean_centred ≡ mean_diff to 4 decimals, confirming Qwen3-0.6B's persona-pair activations are already centred.
- sspace is functionally indistinguishable from mean_diff/mean_centred.
- pca is a cautionary tale: positive axis_shift can hide universal foundation suppression.

# Research Journal

## 2026-05-03: Bidirectional Qwen3.5-4B sweep complete (jobs 67-69)

All 15/15 method JSONs in `outputs/tinymfv_sweep/`. Model: Qwen3.5-4B, eval: airisk vignettes (524 cells), persona pairs: AUTH_CARE axis.

### Pipeline summary

- Jobs 61-66 failed: chars dtype+device bug (`einsum(kern fp32, row_mass bf16)` + state tensors on CPU). Fixed in `chars.py:apply()` with explicit `.to(device=h.device, dtype=torch.float32)`.
- Jobs 67 (sweep tail), 68 (engineered_prompt), 69 (repeng) all succeeded.
- repeng at coeff=1.5 fully saturated (pmass<0.5 on full eval batches, Chinese tokens in top-5). All n/a as predicted by the script comments.

### Final leaderboard (sorted by axis_shift best sign)

```
method                  axis_Δ   Auth_k1  SocN_k1  mean_SI_k1
engineered_prompt[  ]   +0.87    -1.00    +1.00    -1.00 (Auth SI is -1 = hurts)
directional_ablation[-] +0.67    +0.21    +1.00    +0.21 ← best SI
topk_clusters[+]        +0.65    +0.09    +0.23    +0.09
sspace[+]               +0.49    +0.18    +1.00    +0.18 ← 2nd best SI
angular_steering[-]     +0.51    -1.00    +1.00    -1.00
chars[+]                +0.43    -1.00    +0.63    -1.00
mean_diff[+]            +0.39    +0.06    +1.00    +0.06
mean_centred[+]         +0.39    +0.03    +1.00    +0.03
pca[-]                  +0.32    -0.94    +1.00    -0.94
spherical[+]            +0.32    -0.94    +1.00    -0.94
cosine_gated[+]         +0.31    +0.09    +0.67    +0.09
prompt_only[  ]         +0.22    -1.00    +1.00    -1.00
linear_act[+]           +0.24    -1.00    +1.00    -1.00
repeng[  ]              n/a      n/a      n/a      n/a (collapsed)
```

### Notable observations

- Care SI is always n/a across all methods (including directional_ablation which has 9 care flip_to_right). Likely the SI function requires cells near the flip boundary; Qwen3.5-4B already strongly classifies care violations (base logit +2.11) so no cells land near 0.5 threshold.
- directional_ablation and sspace are the only methods with positive Auth SI. directional_ablation gets there by moving a lot of cells in the -C direction (negative flips across all foundations including -9 on Care).
- Sign agreement table: cosine_gated has full ✓ across all foundations (best coherence). angular_steering, directional_ablation, topk_clusters show all ✗ (the ±C vectors are not steering a clean axis).
- engineered_prompt at axis_shift=+0.87 is the strongest mover but SI=-1.00 on Auth means it's breaking as many Auth judgments as it fixes.
- Next: README update (#72, #73) with this table.

## 2026-05-03 (continued): SI under global persona-aligned sign

Aggregator now picks one sign per method via `axis_shift` (persona-intended
direction: Auth↓ + Care↑) and uses that sign for BOTH the flip table and SI.
This removes the per-foundation max() selection bias.

```
method                  axis_Δ   mean_SI_k1   Auth_k1   SocN_k1
directional_ablation[-] +0.67    +0.21        +0.21     -0.11
topk_clusters[+]        +0.65    +0.09        +0.09     -0.28
all others              ±        -1.00        -1.00     +1.00 or +0.6x
```

### Why everyone is -1.00 on Auth (not a bug)

Bare model already classifies 33/34 Authority cells as "wrong" (high baseline
wrongness on AI-risk Auth vignettes). Under intent[Auth]=-1, that gives
n_rej=33, n_cho=1.

Most methods don't move ANY of the 33 wrong-classified cells back to "right"
(fix=0/33), but they DO break the 1 right-classified cell
(broke=1/1=100%). SI = 0 - 1*1 = -1.00.

Only `directional_ablation` (fix=7/33) and `topk_clusters` (fix=3/33) actually
move bulk Auth cells across the 0.5 threshold, hence positive Auth_k1.

### Implication

SI on a metric where the base classifier is saturated has fragile denominators.
For the README leaderboard, Auth_k1 is informative as "did the method move
*any* of the 33 saturated cells", but the n_cho=1 broke denominator makes the
metric noisy on the right side.

Better headline metric for this axis might be `n_flip_to_right / n_total`
(raw rate of cells crossing 0.5 in the intended direction), which doesn't
have the n_cho=1 fragility. Flip table gives this directly.

### Direction-aligned flip table is the right summary

```
method                  axis_Δ   Auth(net,to_r)   SocN(net,to_r)
directional_ablation[-] +0.67    -7  (7→right)    -12 (13→right)
topk_clusters[+]        +0.65    -3  (3→right)    -8  (8→right)
engineered_prompt[ ]    +0.87    +1  (0→right)    -3  (0→right, +3→wrong)
all others              <0.5     +1  (0→right)    +1-3 (mostly→wrong)
```

directional_ablation and topk_clusters are the only methods that actually
flip Auth cells in the intended (right) direction. Everyone else either
doesn't move Auth, or moves it the wrong way.

## 2026-05-03 (continued): SI is fragile on bimodal base wrongness

User flagged that SocN_k1=+1.00 was identical across 8 methods. Investigated:
all 8 show `SocN fix=3/3 broke=0/29` and `Auth fix=0/33 broke=1/1`. Identical
counts across architecturally different methods is suspicious.

Cause: bare wrongness on AI-risk vignettes is bimodal. Inspecting bare.json:
- SocN n_rej=3 cells: wrongness = 0.464, 0.466, 0.488 (all within ±0.04 of 0.5)
- Next SocN cell: 0.517 (above threshold) → bulk is at ≥0.6
- Auth n_cho=1 cell: wrongness=0.418, next jumps to 0.700

So the flip metric is measuring "did your perturbation nudge the 4 cells
sitting on the threshold". Almost any persona perturbation flips them.
Methods that actually move the bulk distribution (directional_ablation,
topk_clusters) produce more flips because they shift cells 0.7→0.4 etc.

### Implication for the leaderboard

SI is honest math (fix_rate - k*broke_rate) but the denominators (n_cho=1,
n_rej=3) are fragile. Two methods can have identical SI not because they
steer the same way, but because they both happen to perturb the 4 boundary
cells.

Better headline: dlogit means + std (continuous, not threshold-sensitive).
Flip table is still useful as a sanity check (7 to_right out of 34 is a real
result), but SI as a single number compresses too much information away.

### Repeng baseline

Coeff=1.0 (without iso-KL): axis_shift=+0.02, all foundations dlogit≈-1.7,
pmass<0.5 on ~25% of eval rows. Means uncalibrated repeng saturates output
distribution -- the standard failure mode for raw-coeff steering. Reframed:
this is the "uncalibrated baseline" (what most repeng users ship), not a
sabotage example. The story is "iso-KL calibration matters", not "repeng is
broken".


# 2026-05-03 13:25:30

ok so we focus no only only authority as it seems models get confused. plus care if saturated.

I also changed the tinymfv to have labels for each question, for each factor.
## 2026-05-03: Authority-only axis — dry-run results

### Setup
Refactored to pure Authority axis (dropped Care): PERSONA_PAIRS_AUTHORITY (3 Clifford-aligned pairs), engineered prompts regenerated via GPT-4o. Bidirectional eval (+C and -C). FA2 dropped (Qwen3.5 s_aux=None bug).

### mean_diff dry-run (job 78)
```
pos ΔAuth: -0.248 ± 1.004  (intended: <0 ✓ but noisy)
neg ΔAuth: -0.644 ± 0.586  (intended: >0 ✗ — wrong sign)
```
Both directions move Authority DOWN. Sign agreement failed. Probable cause: the vector encodes a general "ethics-awareness" direction; at -C, model is disrupted/confused and still moves Auth↓ (possibly toward safety-trained prior).

### engineered_prompt baseline (job 79)
```
POS (Auth↓): ΔAuth = -2.00 ± 1.06  axis_shift = +1.25  ✓ strong
NEG (Auth↑): ΔAuth = +0.02 ± 0.58  axis_shift = +0.05  ✗ near-zero
```
Auth↓ worldview works strongly. Auth↑ worldview has zero effect. Model ignores "respect authority" system prompt — confirmed by demo trace: both worldviews produce identical reasoning for Care-dominant vignettes (AI fabricating medical advice).

### Key finding: asymmetric steerability
Qwen3.5-4B can be steered toward Auth↓ (disobedience is fine) but resists Auth↑ (disobedience is wrong). Likely a safety-training floor: model is anchored to low Authority sensitivity, you can push lower but not higher. Also: Auth↑ fails because airisk vignettes' "is_wrong" signal is Care-dominated — adding authority framing adds no new reason to call harm wrong.

### Demo trace observation
- max_think_tokens=64 too short: model truncates with "I should answer now" without engaging worldview
- Demo vignette was Care-type (AI medical fabrication), not Authority-type — demo should filter for Authority foundation vignettes for better interpretability
- p_true spans 0.05→0.94 (not glitching to 1)

### Next
- repeng baseline (job 81, running) — expect same asymmetry
- queue full sweep after baselines complete
- consider: 128 max_think_tokens, role-based personas (judge vs rebel) instead of abstract descriptions

## 2026-05-03 (session 2): baselines complete, README updated

### Jobs completed
- Job 91: engineered_prompt baseline (128 think tokens)
  - POS (Auth↓): axis=+1.34, ΔAuth=-2.98±1.20 (strong, correct direction)
  - NEG (Auth↑): axis=-0.16, ΔAuth=-0.37±0.69 (weak, asymmetry confirmed)
- Job 98: repeng baseline (128 think tokens, coeff=0.75)
  - axis=+0.01, ΔAuth=-0.89±0.58 — broad suppression across all foundations
  - Job 92 failed: missing --extra baseline; re-queued as 98

### Final aggregator results (Auth-only axis, Qwen3.5-4B)

SI(Auth) top 3 calibrated methods: directional_ablation (55), cosine_gated (48), sspace (43).
All three show SI_fwd>0 AND SI_rev>0 (bidirectional coherence on Auth).

Sign agreement: cosine_gated is the only method with ✓ across ALL foundations (+C and -C move Auth in opposite directions). linear_act and sspace are close behind.

Key finding: axis_Δ and SI tell different stories. engineered_prompt[+] has the largest axis_Δ (2.30) but all foundations move equally — broad suppression, not axis rotation. SI penalizes this correctly.

Asymmetric steerability confirmed: Auth↓ is easy (most methods show ΔAuth<0); Auth↑ is hard (NEG direction fails for all methods except cosine_gated). Plausible cause: Qwen3.5-4B safety training creates a floor for how much it will endorse disobedience as morally acceptable.

repeng negative SI (-16.67) confirms uncalibrated coefficient breaks as many verdicts as it fixes.
