# Research Journal — steering-lite

## 2026-05-03 — Full bidirectional sweep, Qwen3.5-4B (job 87)

Model: `Qwen/Qwen3.5-4B`, 128 think tokens, Authority axis, bidirectional eval.
Run ID: `40079d54d4f4`, timestamp: `2026-05-03T14:07`.

**Note**: `sspace` here is OLD activation-SVD on residual (pre-refactor). New weight-SVD sspace is job 118 (queued).

```
cue   axis  row                          C_calib  kl_p95  ΔCare      ΔSanc      ΔAuth      ΔLoy       ΔFair      ΔLib       ΔSocN      t
⚪    ref   bare                         n/a      n/a     +2.38±0.72 +2.41±0.77 +2.56±0.74 +2.31±0.72 +1.80±1.15 +2.63±0.69 +1.52±1.12 277s
🔴    +0.02 prompt_only                  n/a      n/a     -2.01±1.51 -2.55±1.42 -2.03±1.38 -2.07±1.54 -1.99±1.66 -2.93±1.40 -1.82±1.77 291s
🟢    +0.52 steer_mean_diff[+]           +0.606   0.97    -0.64±1.38 -0.82±1.46 -1.16±1.57 -1.15±1.62 -1.38±1.50 -0.72±1.45 -2.17±1.47 671s
🟡    +0.48 steer_mean_centred[+]        +0.598   0.96    -0.63±1.37 -1.06±1.60 -1.11±1.54 -1.11±1.35 -1.12±1.13 -0.43±1.51 -2.13±1.43 679s
🟡    +0.48 steer_pca[+]                 +0.660   0.70    -1.03±0.83 -1.06±0.73 -1.51±1.04 -0.94±0.65 -0.83±1.00 -0.76±0.81 -0.93±1.03 717s
🟢    +0.66 steer_topk_clusters[+]       +0.733   1.00    +0.71±1.27 +0.27±1.80 +0.05±1.53 +0.07±1.84 -0.01±1.83 +0.70±1.42 -1.37±1.52 746s
🟢    +1.04 steer_cosine_gated[+]        +5.126   1.45    -0.27±1.00 -0.66±1.10 -1.32±1.06 -1.10±1.01 -0.98±1.01 -0.73±1.19 -1.85±1.01 772s
🟡    +0.48 steer_sspace[+] (OLD)        +0.597   1.04    -0.61±1.24 -1.40±1.45 -1.09±1.18 -1.31±1.39 -1.10±1.25 -1.10±1.60 -2.14±1.09 666s
🟡    +0.25 steer_spherical[+]           +0.039   1.04    -0.56±1.08 -0.85±1.10 -0.82±1.06 -1.21±1.12 -1.01±1.18 -0.70±1.10 -1.91±1.15 722s
🟡    +0.50 steer_directional_ablation[-]-1.541   1.03    -0.32±0.53 -0.15±0.77 -0.82±0.73 -0.49±0.58 -0.29±0.74 -0.49±0.76 +0.50±0.74 681s
🟡    +0.43 steer_chars[-]               -0.788   0.99    +0.12±0.50 +0.02±0.61 -0.31±0.72 -0.18±0.68 +0.09±0.73 -0.02±0.40 +0.11±0.91 746s
🟡    +0.36 steer_linear_act[+]          +2.235   1.03    -0.04±0.68 +0.03±0.60 -0.40±0.79 -0.27±0.76 -0.44±0.86 -0.13±0.60 -0.95±1.34 696s
🟢    +0.53 steer_angular_steering[-]    -0.625   0.95    -1.30±0.63 -1.38±0.64 -1.83±0.72 -1.59±0.69 -1.24±0.88 -1.57±0.61 -1.01±0.67 676s
```

**Δlogit observations** (raw sweep table):
- `angular_steering` largest raw ΔAuth (-1.83) with lowest std — looks best on Δlogit alone.
- `mean_diff` best overall green cue at kl_p95≈1.0.
- `cosine_gated` kl=1.45 (over-budget), C_calib=5.126.
- OLD `sspace` (activation-SVD): ΔAuth -1.09, comparable to mean_diff. Job 118 (new weight-SVD) pending.

**SI (Surgical Informedness) via aggregate_flips.py** — headline metric, sorted by SI(Auth):

```
method                    SI(Auth)  SI_fwd  SI_rev  axis_Δ  sign_agree(all_f)
directional_ablation[+]    55.90    0.41    +0.99   2.14    ✗ (same sign both arms)
cosine_gated[+]            50.40    0.28    +1.00   1.32    ✓ (bidirectional all f)
sspace[+] (OLD)            44.16    0.27    +1.00   1.47    ✗
mean_centred[+]            37.57    0.25    +1.00   1.44    ✗
mean_diff[+]               37.15    0.26    +1.00   1.48    ✗
linear_act[+]              28.14    0.08    +1.00   0.60    ✗
engineered_prompt[+]       17.36    0.50    -0.02   2.52    ✗ (one arm broken)
repeng                      9.02    0.10    n/a     1.10    n/a
spherical[+]                6.73    0.23    n/a     1.30    ✗
chars[-]                    1.38   -0.23    +0.26   0.30    ✓
gated_subspace_ablation[+] -7.80   -0.73    +0.50   2.26    ✗ (fwd arm wrong dir)
pca[+]                     -9.16   -0.18    -0.11   1.44    ✗ (both arms wrong)
angular_steering[+]       -25.35   -0.55    -0.11   2.47    ✗ (BOTH ARMS WRONG)
topk_clusters[-]          -46.31   -0.99    -0.16   0.31    ✗ (very wrong)
```

**Critical finding**: `angular_steering` has largest raw Δlogit (2.47) but SI=-25.35 — both forward and reverse arms move opposite to intent. The raw Δlogit metric is misleading for this method. `directional_ablation` and `cosine_gated` lead on the informedness metric.

`cosine_gated` is the only method with full bidirectional sign agreement across all 7 foundations AND positive SI. Old `sspace` has SI_fwd=0.27 (very low) vs SI_rev=1.00 — asymmetric arms.

**Baseline for new sspace comparison**: mean_diff SI=37.15, axis_Δ=1.48, kl=0.97. Target: SI>37 with SI_fwd>0.27.

---

## 2026-05-03 — sspace refactor committed (caf6ad0)

- Replaced old activation-SVD sspace with AntiPaSTO weight-SVD cosine-gated in S-space.
- Added `sspace_ablate` (projection-out in S-space).
- Removed `requires_linear_io` flag; dispatch on `cfg.target_submodule is not None`.
- Deleted `gated_subspace_ablation.py` (was misinterpretation of same idea).
- Smoke: 11/11 passing.
- New sweep job 118 queued (behind weight-steering jobs 113-117).

**Hypothesis for job 118**: New sspace shows ΔAuth comparable or larger than mean_diff (-1.16) at kl≈1.0, with non-degenerate gate values (cos in r=8 S-space should be meaningful per-token signal). sspace_ablate at coeff=0 shows non-zero ΔAuth iff contrastive direction in down_proj S-space is load-bearing.

---

## 2026-05-04 — Job 119: new weight-SVD sspace + sspace_ablate (bidirectional sweep, Qwen3.5-4B)

Job 118 failed (CUDA device mismatch: weight SVD on GPU, activations on CPU). Fixed in `sspace.py`: `W.detach().cpu()` and `mod.bias.detach().cpu()`. Re-ran as job 119, completed 2026-05-04T04:22 AWST.

Model: `Qwen/Qwen3.5-4B`, 256 think tokens, Authority axis. Run ID: `4c338a356760`.

```
cue  	axis  	row                          	C_calib  	kl_p95  	ΔCare     	ΔSanc     	ΔAuth     	ΔLoy      	ΔFair     	ΔLib      	ΔSocN     	t
⚪    	ref   	bare                         	n/a      	n/a     	+2.55±0.55	+2.59±0.59	+2.74±0.35	+2.59±0.45	+2.15±1.25	+2.77±0.51	+1.85±1.29	467s
🟡    	+0.41 	prompt_only                  	n/a      	n/a     	-1.96±1.62	-2.19±1.63	-2.36±1.54	-2.26±1.50	-2.35±1.66	-2.90±1.47	-1.90±1.98	466s
🟢    	+0.54 	steer_sspace[+]              	+2.476   	1.04    	-2.43±0.93	-2.72±0.79	-2.97±0.57	-2.85±0.63	-2.46±1.26	-2.58±0.81	-2.45±1.45	1324s
🟢    	+0.55 	steer_sspace_ablate[-]       	-0.847   	1.01    	-2.15±0.56	-2.27±0.52	-2.70±0.52	-2.42±0.53	-1.92±1.13	-2.27±0.48	-1.71±1.10	1291s
🟢    	+0.54 	steer_mean_diff[+]           	+0.673   	0.99    	-1.50±1.00	-1.83±1.19	-2.04±1.15	-1.63±1.05	-1.93±1.01	-1.37±0.79	-2.26±1.38	1104s
🟡    	+0.30 	steer_mean_centred[+]        	+0.662   	0.95    	-1.51±1.13	-1.80±1.17	-1.81±1.25	-1.38±1.07	-1.86±1.13	-1.20±0.88	-2.13±1.21	1098s
🟡    	+0.48 	steer_pca[+]                 	+0.687   	1.05    	-1.27±1.36	-1.25±1.44	-1.75±1.70	-1.38±1.39	-0.75±1.59	-1.21±1.40	-1.38±1.34	1155s
🟡    	+0.49 	steer_topk_clusters[-]       	-0.799   	1.03    	+0.66±0.53	+0.37±0.73	+0.18±0.79	+0.06±0.75	+0.26±0.77	+0.04±0.75	+0.95±0.80	1156s
🟢    	+0.63 	steer_cosine_gated[+]        	+6.887   	1.04    	-1.82±0.57	-1.69±0.61	-2.45±0.35	-2.15±0.48	-1.67±1.08	-1.96±0.49	-1.59±1.08	1232s
🟡    	+0.42 	steer_spherical[+]           	+0.038   	1.01    	-1.08±0.63	-1.33±0.85	-1.50±0.71	-1.41±0.75	-1.14±0.92	-1.00±0.38	-1.79±1.31	1215s
🟡    	+0.34 	steer_directional_ablation[+]	+1.429   	0.95    	-1.64±1.24	-2.21±1.35	-1.97±1.29	-1.70±1.25	-2.05±0.98	-1.62±1.30	-2.02±1.19	1156s
🟡    	+0.31 	steer_chars[-]               	-0.774   	1.01    	-0.27±0.58	-0.54±0.42	-0.58±0.37	-0.38±0.42	-0.21±0.92	-0.59±0.32	-0.44±0.88	1289s
🟡    	+0.25 	steer_linear_act[-]          	-2.356   	1.02    	+0.09±0.57	+0.04±0.65	-0.16±0.39	+0.10±0.37	+0.41±1.10	-0.27±0.42	+0.46±0.94	1144s
🟢    	+0.54 	steer_angular_steering[+]    	+0.326   	1.03    	-2.46±0.72	-2.47±0.85	-2.99±0.81	-2.74±0.88	-2.13±1.21	-2.70±0.85	-2.20±1.24	1149s
```

**SI (Surgical Informedness) via aggregate_flips.py** — headline metric, sorted by SI(Auth):

```
method                    SI(Auth)  SI_fwd  SI_rev  axis_Δ  Auth_sep  pmass²×100
directional_ablation[+]    52.90    0.32    +1.00   1.94    +2.05     80.1
sspace[+]                  45.67    0.64    +0.85   2.78    +0.69     61.0
mean_diff[+]               32.81    0.34    +1.00   1.93    +1.65     49.0
mean_centred[+]            32.72    0.29    +1.00   1.80    +1.56     50.6
topk_clusters[+]           31.34    0.13    +0.72   1.18    +1.55     73.9
sspace_ablate[+]           24.11    0.74    +0.02   2.89    +0.59     63.6
cosine_gated[+]             8.92    0.09    +1.00   2.08    +2.00     16.4
angular_steering[+]         7.00    0.55    -0.38   2.67    +0.32     80.6
pca[+]                     -0.92    0.03    -0.08   1.36    +0.85     39.0
gated_subspace_ablation[+] -7.80   -0.73    +0.50   n/a     +0.17     69.2
chars[-]                   -9.16   -0.26    -0.00   0.45    +0.50     68.3
```

**Key findings**:

- **New sspace SI=45.67** — beats mean_diff (32.81) and old sspace (44.16). Hypothesis confirmed.
- **SI_fwd=0.64 vs 0.27 (old)** — forward arm more than doubled. Weight-SVD cosine gate is doing real work, not just adding a bias.
- **sspace raw ΔAuth=-2.97**, largest among non-angular methods. axis_Δ=2.78 (loading-weighted), second only to sspace_ablate.
- **sspace_ablate axis_Δ=2.89** — highest raw shift (apart from prompt_only), confirming down_proj S-space contrastive direction is load-bearing. But SI_rev=+0.02 — ablation has no reverse arm by construction, so SI is lower despite strong forward effect. SI_fwd=0.74 (best forward).
- **cosine_gated SI dropped** from 50.4 (job 87) to 8.92 here. pmass²=16.4 (very low probability mass on top answers) — model is uncertain/spread out. The gate helps Δlogit but hurts SI because it over-concentrates rather than redirecting.
- **angular_steering SI=7.0** — much improved from -25.35 (job 87), but still one-sided (SI_rev=-0.38). The raw axis_Δ=2.67 is again misleadingly large.

**sspace vs sspace_ablate (both hook down_proj, same S-space)**:
- sspace (cosine-gated additive): SI=45.67, SI_fwd=0.64, SI_rev=0.85 — bidirectional informedness
- sspace_ablate (projection-out): SI=24.11, SI_fwd=0.74, SI_rev=0.02 — strong one-way ablation but no reverse arm

This confirms the contrastive direction in down_proj S-space is structurally important (not just a correlation artifact): projecting it out has as strong an effect on logits as the best additive methods.

## 2026-05-04 — job 125 sweep #1 (residual writers, r=−1, multi-submodule refactor)

Job 125 (run_id=99f38378522e, model=Qwen3-0.6B, target=mlp.down_proj+self_attn.o_proj per block via regex, r=−1 full rank). Note: super_sspace was missing from METHODS at job-start time; appears in job 126.

### Final sweep table (axis_shift = −ΔAuth − ΔCare; bare row absolute, others paired Δ vs bare)

```
cue  axis  row                            C_calib   kl_p95  ΔCare       ΔSanc       ΔAuth       ΔLoy        ΔFair       ΔLib        ΔSocN       t
⚪    ref   bare                            n/a       n/a     +0.98±2.38  -0.21±2.38  +0.99±2.53  +0.15±2.15  +0.67±2.37  +1.69±2.11  -1.46±2.16  240s
🟡   +0.41 prompt_only                     n/a       n/a     -1.68±2.92  -1.45±2.49  -2.09±3.39  -2.03±2.69  -2.28±2.64  -2.00±2.27  -0.41±2.66  245s
🟡   +0.36 steer_sspace[-]                 -25.600   0.55    -2.28±2.61  -1.67±2.63  -2.63±2.73  -1.83±2.19  -2.17±2.26  -2.66±2.35  -0.28±2.51  995s
🟡   -0.17 steer_sspace_ablate[-]          -4.178    0.95    -1.53±2.35  -0.49±2.34  -1.36±2.49  -0.61±2.05  -1.13±2.37  -2.12±2.18  +0.86±2.21  968s
🔴   +0.05 steer_sspace_damp_amp[+]        +6.646    1.00    -1.29±2.41  -0.21±2.48  -1.34±2.49  -0.49±2.10  -1.05±2.29  -2.02±2.20  +1.04±2.19  852s
🔴   -0.15 steer_mean_diff[+]              +2.169    0.98    -1.97±2.74  -1.26±2.51  -1.83±2.32  -1.58±2.57  -1.66±2.42  -2.68±1.77  +0.23±2.42  443s
🔴   -0.15 steer_mean_centred[+]           +2.169    0.98    -1.97±2.74  -1.26±2.51  -1.83±2.32  -1.58±2.57  -1.66±2.42  -2.68±1.77  +0.23±2.42  442s
🟡   +0.33 steer_pca[-]                    -2.473    1.03    -0.80±2.26  -0.68±1.88  -1.13±1.90  -0.48±2.25  -0.80±1.98  -1.45±2.10  +0.34±2.21  465s
🟡   -0.27 steer_topk_clusters[+]          +2.582    1.01    -3.55±3.13  -2.28±3.11  -3.28±3.60  -2.96±2.96  -3.29±3.28  -3.72±2.92  -1.51±2.89  569s
🟢   +0.71 steer_cosine_gated[-]           -12.004   1.04    +0.54±2.16  +1.04±2.14  -0.17±2.31  +0.67±2.00  +0.25±1.94  -0.11±2.12  +1.47±2.29  481s
🟢   +0.99 steer_spherical[+]              +0.013    1.00    -1.43±3.12  -1.36±2.92  -2.43±3.12  -0.90±3.04  -1.36±2.35  -2.56±2.79  -0.78±2.44  538s
⚪   +nan  steer_directional_ablation[-]   -0.006    15.29   -0.05±0.00  n/a         n/a         n/a         -0.35±1.64  n/a         n/a         567s
🟡   +0.21 steer_chars[+]                  +2.467    1.04    -0.98±2.28  +0.02±2.31  -1.18±2.53  -0.00±2.13  -0.59±2.35  -1.72±2.14  +1.25±2.11  662s
🟡   +0.35 steer_linear_act[+]             +2.632    0.97    -1.38±2.51  -1.15±2.37  -1.74±2.85  -1.34±2.48  -1.75±2.95  -2.06±2.67  -0.91±2.23  485s
⚪   +nan  steer_angular_steering[-]       -0.006    16.21   n/a         n/a         n/a         n/a         n/a         n/a         n/a         593s
```

### SI(Auth) ranking (loading-weighted, k=1.0, headline=Authority)

```
method                 SI(Auth)  SI_fwd  SI_rev  Care_SI  Auth_sep  pmass²×100
linear_act[+]          +61.39    +0.44   +0.85   -52.44   +2.42     94.8
mean_centred[+]        +55.66    +0.54   +0.73   -54.02   +1.90     88.0
mean_diff[+]           +55.66    +0.54   +0.73   -54.02   +1.90     88.0
sspace_ablate[-]       +54.52    +0.94   +0.30   -44.52   +0.63     87.6
sspace[-]              +53.40    +0.94   +0.21   -43.12   +1.62     93.5
topk_clusters[+]       +49.72    +0.89   +0.17   -42.44   +2.72     93.6
spherical[+]           +43.67    +0.53   +0.67   -36.68   +2.05     73.1
prompt_only            +39.12    +0.40   n/a     -29.48   n/a       97.5
cosine_gated[+]        +38.08    +0.61   +0.27   -38.31   +1.90     86.6
chars[+]               +26.45    +0.13   +0.52   -20.34   +0.71     81.0
sspace_damp_amp[+]     +19.63    +0.59   -0.11   -20.26   +0.33     81.5
directional_ablation[+] +1.52    n/a     +0.57    -1.09   +0.00     2.7
pca[-]                 -3.93     +0.09   -0.19   +8.61    +0.03     86.1
angular_steering[+]    n/a       n/a     n/a     n/a      n/a       0.0
```

### Observations

- **sspace 45.67 → 53.40** (+7.7 SI). Task-specific |dS|.topk replaced top-σ truncation; full-rank r=−1. Modest improvement.
- **sspace_ablate +54.52** beats sspace by ~1 SI now (was 24 vs 45 before). The new shared extract (full SVD + |dS|.topk) helps ablation more than additive — projection-out is more sensitive to the basis being task-relevant.
- **C_calib=−25.6, kl_p95=0.549** for sspace: at full rank the d-space cosine gate concentrates near 0; calibration scaled α 10× higher than mean_diff and still stopped at half target_kl. The gate being near-constant explains why sspace doesn't dominate despite large α.
- **sspace_damp_amp +19.63**: weak. The multiplicative `(exp(c·dS_hat)−1)` mass at full rank is tiny; would need much larger α than calibrator allowed. Probably wants r << d_model.
- **mean_diff ≡ mean_centred** (byte-identical): `subtract_corpus_mean=True` is no-op in the current impl. Pre-existing bug, not from this refactor.
- **angular_steering / directional_ablation NaN**: pmass collapse — both methods produced near-zero coeffs (C≈±0.006) and degenerate distributions. Existing failure, untouched here.
- **Sign agreement**: only mean_diff/cosine_gated/linear_act/spherical pass on every foundation. sspace*, topk_clusters fail on most — the high SI is from the chosen-sign arm; reverse arm doesn't move the axis.

### Next

- Job 126 (all 7 Linears, full METHODS list with super_sspace) running now — will see whether multi-submodule helps and whether super_sspace's pooled basis beats per-Linear sspace.
- Job 127 (super_sspace + sspace + mean_diff at residual-stream hook) queued after.

## 2026-05-04 — job 126 sweep #2 (all-Linear, r=−1) + job 127 super_sspace test

Job 126 (run_id=6cb19b7b2641): same as job 125 but with `target_submodule=r"self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"` — hooks all 7 Linears per block instead of just the 2 residual writers. 14 methods including super_sspace.

Job 127 (run_id=fb76ad999134): 3-method test (super_sspace, sspace, mean_diff). super_sspace ignores `--sspace-target-submodule` (uses model walk via shape detection) so its result matches 126's exactly — a sanity check, not new data.

### Job 126 SI(Auth) ranking

```
method                  SI(Auth)  SI_fwd  SI_rev  Care_SI  Auth_sep  pmass²×100
linear_act[+]           +61.39    +0.44   +0.85   -52.44   +2.42     94.8
mean_centred[+]         +55.66    +0.54   +0.73   -54.02   +1.90     88.0
mean_diff[+]            +55.66    +0.54   +0.73   -54.02   +1.90     88.0
topk_clusters[+]        +49.72    +0.89   +0.17   -42.44   +2.72     93.6
super_sspace[+]         +47.71    +0.67   +0.40   -45.99   +1.99     88.8
spherical[+]            +43.67    +0.53   +0.67   -36.68   +2.05     73.1
prompt_only             +39.12    +0.40   n/a     -29.48   n/a       97.5
cosine_gated[+]         +38.08    +0.61   +0.27   -38.31   +1.90     86.6
sspace[-]               +33.54    +0.85   -0.06   -23.53   +1.15     85.3
sspace_ablate[-]        +32.93    +0.63   +0.13   -26.78   +0.21     87.0
chars[+]                +26.45    +0.13   +0.52   -20.34   +0.71     81.0
sspace_damp_amp[+]      +22.69    +0.64   -0.14   -32.96   +0.48     90.5
directional_ablation[+] +1.52     n/a     +0.57    -1.09   +0.00     2.7
pca[-]                  -3.93     +0.09   -0.19   +8.61    +0.03     86.1
angular_steering[+]     n/a       n/a     n/a     n/a      n/a       0.0
```

### Cross-sweep comparison: writers-only vs all-Linear vs super_sspace

| method | writers-only (job 125) | all-Linear (job 126) | residual-stream pooled |
|---|---|---|---|
| sspace | **+53.40** | +33.54 | — |
| sspace_ablate | **+54.52** | +32.93 | — |
| sspace_damp_amp | +19.63 | +22.69 | — |
| **super_sspace** | (missing) | **+47.71** | +47.71 (same) |

### Key observations

- **Hooking more Linears HURTS sspace and sspace_ablate** (53→33, 54→33). Hypothesis: with 7 hooks injecting independent additive deltas per block, the iso-KL budget gets spread thin per submodule; each submodule's local steering is weaker, and the ablation/additive directions stop being task-coherent across submodules.
- **super_sspace dominates the sspace family at all-Linear** (+47.71 vs +33.54). Pooling the basis into one residual-stream hook is *more* effective than spreading the same KL budget across 7 per-Linear hooks. Confirms the hypothesis from the variant's docstring: a global d_model basis is more robust than tail-noisy per-Linear bases.
- **super_sspace is 4× cheaper than per-Linear sspace** (582s vs 2279s in job 126) — single block-level hook, basis precomputed once via Gram trick.
- **Cleaner calibration**: super_sspace C=−3.9 kl=1.02 (saturated); sspace at writers-only C=−25.6 kl=0.55 (under-saturated, suggesting near-dead cosine gate at full rank in the per-Linear case).
- **mean_diff still tops the SI table** (+55.66) regardless of regime — input-independent additive is hard to beat at iso-KL=1.0 on this benchmark. The sspace-family advantage will likely show at higher KL targets where additive saturates and gating starts paying off.

### What's pending

- README update with cross-method/cross-target table.
- r=64 / r=128 sweep at writers-only to test whether moderate-rank cropping recovers the cosine-gate (super_sspace at smaller r should also be tried).
- Investigate `mean_diff ≡ mean_centred` byte-identical bug (`subtract_corpus_mean` no-op) — pre-existing, separate from this work.

## 2026-05-04 — job 128 r=64 writers-only follow-up

Job 128 (run_id pending). Same as job 125 (writers-only target_submodule) but `--sspace-r 64` instead of `-1`. 6 methods: sspace family + super_sspace + mean_diff/cosine_gated controls.

### SI(Auth) ranking (r=64 writers-only)

```
method             SI(Auth)  SI_fwd  SI_rev  Care_SI  Auth_sep  pmass²×100
mean_diff[+]       +55.66    +0.54   +0.73   -54.02   +1.90     88.0
sspace[-]          +47.75    +1.00   +0.01   -52.68   +2.28     95.0
sspace_ablate[-]   +41.99    +0.68   +0.20   -25.97   +0.95     95.5
prompt_only        +39.12    +0.40   n/a     -29.48   n/a       97.5
cosine_gated[+]    +38.08    +0.61   +0.27   -38.31   +1.90     86.6
sspace_damp_amp[+] +24.72    +0.59   +0.03   -26.10   +0.40     80.3
super_sspace[+]    +0.69     -0.16   +0.18   +2.84    +0.27     80.3
```

### r vs SI head-to-head (writers-only)

| method | r=−1 SI | r=64 SI | r=−1 ax_max | r=64 ax_max |
|---|---|---|---|---|
| mean_diff | 55.66 | 55.66 | −0.15 | −0.15 |
| sspace | **53.40** | 47.75 | +0.36 | +0.48 |
| sspace_ablate | **54.52** | 41.99 | −0.17 | +0.29 |
| sspace_damp_amp | 19.63 | **24.72** | +0.05 | −0.14 |
| super_sspace (alllin) | **47.71** | 0.69 | +0.96 | +0.17 |
| cosine_gated | 38.08 | 38.08 | +0.71 | +0.71 |

### Surprise: SI ↓ but axis_shift ↑ for sspace at r=64

axis_shift (one-sign max) and SI (bidirectional informedness) disagree:
- r=64: SI_fwd=1.00, SI_rev=0.01. The gate pushes hard one way and is *dead* the other way.
- r=−1: SI_fwd=0.94, SI_rev=0.21. Weaker per-direction but bidirectional.

**Interpretation:** at r=64 the task-specific top-r selection picks modes maximally aligned with the contrastive direction. The cosine gate `|cos(xS, dS_hat)|` is sign-agnostic in magnitude, but `dS_hat` is fixed — so positive-α steers along `+dS`, negative-α along `−dS`. At r=64, the steering signal is concentrated on a few r-dim modes that the model uses asymmetrically (one direction is "natural", the other isn't). At r=−1, with the noisy d-dim basis, both directions hit a roughly symmetric noise floor.

**For SI (bidirectional metric)**: r=−1 wins — broader basis = more symmetric pushability.
**For raw axis_shift (one-sign)**: r=64 wins — concentrated basis = stronger per-direction movement.
**Practical:** if you only care about pushing one way (e.g. authority↓), r=64 is better. If you want a steering axis usable in both directions, r=−1 is better.

### super_sspace collapses at r=64

SI dropped 47.71 → 0.69 (essentially dead). axis_shift +0.96 → +0.17. Calibration kl=0.85 (didn't saturate at C=2.79).

**Interpretation:** super_sspace's strength comes from the *pooled* full d_model basis — averaging tail noise across many writers/readers leaves clean signal in *all* modes. Cropping to top-64 by |dS| discards most of that signal. Per-Linear sspace benefits from cropping (per-Linear tail is noisy); super_sspace does not (pooled tail is clean).

This is a strong claim about the geometry: per-Linear bases are noisy, the noise mostly lives in low-σ modes, and pooling averages it out. Suggests an opposite cropping rule for super_sspace: drop *zero-eigenvalue tail* of the Gram matrix (modes with no signal) but keep the rest. Currently `_build_super_basis` returns the full d_model spectrum; would want to threshold by σ_⋆ rather than crop by |dS|.topk.

### Final ranking (best of all sweeps)

```
method                    best regime         SI(Auth)
linear_act                alllin r=−1         +61.39  (untouched by this work)
mean_diff/mean_centred    any                 +55.66  (untouched)
sspace_ablate             writers r=−1        +54.52
sspace                    writers r=−1        +53.40
topk_clusters             alllin/writers      +49.72
super_sspace              alllin r=−1         +47.71  ← new variant; cheapest of the gated family
spherical                 any                 +43.67
prompt_only               n/a                 +39.12
cosine_gated              any                 +38.08
```

super_sspace's SI is mid-pack but its **wallclock is 4× faster** than per-Linear sspace (582s vs 2279s). For iterative experimentation it's the best bang/buck of the sspace family.


# 2026-05-04 18:26:04

● Results summary

  4 sweeps run on Qwen3-0.6B, tinymfv airisk benchmark, iso-KL@1.0:

  ┌───────┬─────────────────────────┬─────┬──────────────────┬───────────┬──────────────────┐
  │ sweep │         regime          │  r  │ super_sspace SI  │ sspace SI │ sspace_ablate SI │
  ├───────┼─────────────────────────┼─────┼──────────────────┼───────────┼──────────────────┤
  │ 125   │ writers (down + o_proj) │ −1  │ (missing)        │ 53.40     │ 54.52            │
  ├───────┼─────────────────────────┼─────┼──────────────────┼───────────┼──────────────────┤
  │ 126   │ all 7 Linears           │ −1  │ 47.71            │ 33.54     │ 32.93            │
  ├───────┼─────────────────────────┼─────┼──────────────────┼───────────┼──────────────────┤
  │ 127   │ super_sspace dedicated  │ −1  │ 47.71            │ 53.40     │ —                │
  ├───────┼─────────────────────────┼─────┼──────────────────┼───────────┼──────────────────┤
  │ 128   │ writers                 │ 64  │ 0.69 (collapsed) │ 47.75     │ 41.99            │
  └───────┴─────────────────────────┴─────┴──────────────────┴───────────┴──────────────────┘

  Headline findings:

  1. super_sspace works. SI=+47.71 at all-Linear/r=−1 in 582s (4× faster than per-Linear sspace at 2279s). Single residual-stream hook, pooled Gram-trick basis. New variant earns its place.
  2. Optimal regime differs by variant.
    - per-Linear sspace: writers-only + r=−1 best for SI (53.40); r=64 wins on raw axis_shift but kills bidirectionality (SI_rev → 0.01).
    - super_sspace: all-Linear / full-rank only. r=64 collapses (SI=0.69). Pooled basis needs full d_model.
  3. Hooking more Linears hurts per-Linear sspace (53.40 → 33.54). KL budget gets spread thin per submodule.
  4. mean_diff still tops the SI table at +55.66 — input-independent additive is hard to beat at iso-KL=1.0.
  5. 