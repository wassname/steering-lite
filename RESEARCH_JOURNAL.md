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
