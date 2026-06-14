# Research Journal — steering-lite

## 2026-05-12 — Full 4B tinymfv forced-choice sweep (pueue 90, Qwen3-4B, run_id=c7b02f03306f)

Model: Qwen/Qwen3-4B (Qwen3.5-4B failed: hybrid/linear-attn incompatible with KV-fork in guided_rollout_forced_choice). Eval: 7-way forced-choice, classic vignettes (264), target_kl=0.50, max_think=256. Previous 4B sweep (run_id=4c338a356760) used old binary is_wrong eval on Qwen3.5-4B — numbers not comparable.

### SI table (Auth↓ intent, Qwen3-4B, forced-choice; corrected sign flip)

Sign in brackets = selected direction (whichever of [+]/[-] achieves ΔAuth↓). First pass used [+] always; corrected by selecting the direction with lower ΔAuth as pos_report.

| method | SI(Auth)×100 | SI_fwd | SI_rev | Auth_sep | kl_p95 |
|---|---:|---:|---:|---:|---:|
| pca[-] | +36.56 | +0.13 | +0.60 | -0.84 | 0.49 |
| linear_act[+] | +34.95 | +0.03 | +0.67 | -0.64 | 0.52 |
| spherical[+] | +16.67 | +0.06 | +0.27 | -0.43 | 0.75 |
| sspace_ablate[+] | +15.05 | +0.03 | +0.27 | -0.62 | 0.47 |
| topk_clusters[-] | +3.23 | +0.06 | +0.00 | -0.13 | 0.49 |
| super_sspace[+] | +1.61 | +0.03 | +0.00 | -0.55 | 0.51 |
| chars[+] | +0.00 | +0.00 | +0.00 | -0.44 | 0.46 |
| mean_centred[+] | -3.23 | +0.00 | -0.06 | -0.59 | 0.47 |
| mean_diff[+] | -3.23 | +0.06 | -0.13 | -0.59 | 0.50 |
| cosine_gated[+] | -6.45 | +0.00 | -0.13 | -0.29 | 0.48 |
| angular_steering[-] | -24.73 | -0.63 | +0.14 | -0.59 | 3.48 |
| sspace[+] | -26.34 | -0.67 | +0.14 | -0.16 | 0.49 |
| directional_ablation[+] | -36.56 | -0.67 | -0.06 | +0.16 | 1.59 |
| sspace_damp_amp[+] | -39.78 | -0.60 | -0.19 | -0.11 | 0.51 |
| prompt_only[+] | -126.88 | -1.27 | n/a | n/a | n/a |

### Key findings

1. **pca[-] is the top performer** (SI=+36.56) when the correct sign is selected. The [-] direction moves ΔAuth=-0.84, stronger than any other method. In first-pass (always [+]), pca showed SI=-77.96 — a 113-point swing from sign selection alone.

2. **5 methods have positive SI**, all well-calibrated (kl_p95 ≈ 0.47–0.75): pca[-], linear_act[+], spherical[+], sspace_ablate[+], topk_clusters[-].

3. **Calibration failures for angular_steering and directional_ablation.** Both overshoot target KL badly (3.48 and 1.59 vs target 0.50). angular_steering moves Auth strongly (ΔAuth=-0.59) but is too noisy to rank well on SI.

4. **prompt_only strongly moves Auth in the wrong direction** (ΔAuth=+1.80, SI=-126.88). The Care↑ persona prompt causes Qwen3-4B to become MORE authority-respecting in forced-choice. Model-specific persona effect, opposite to old Qwen3.5-4B result.

5. **Method rankings not portable across models or eval formats.** Old Qwen3-0.6B top-3 were linear_act, mean_diff/mean_centred, sspace_ablate. On Qwen3-4B the ordering shifts (pca rises, sspace methods split by sign).

### Δlogit table (selected sign, sorted by ΔAuth ascending)

| method | axis | ΔCare | ΔAuth | kl_p95 |
|---|---:|---:|---:|---:|
| pca[-] | +1.40 | +0.56 | -0.84 | 0.49 |
| linear_act[+] | +1.27 | +0.63 | -0.64 | 0.52 |
| sspace_ablate[+] | +0.79 | +0.17 | -0.62 | 0.47 |
| angular_steering[-] | +1.47 | +0.88 | -0.59 | 3.48 |
| mean_centred[+] | +0.77 | +0.18 | -0.59 | 0.47 |
| mean_diff[+] | +0.92 | +0.34 | -0.59 | 0.50 |
| super_sspace[+] | +0.58 | +0.02 | -0.55 | 0.51 |
| chars[+] | +0.71 | +0.26 | -0.44 | 0.46 |
| spherical[+] | +0.42 | -0.01 | -0.43 | 0.75 |
| cosine_gated[+] | +0.24 | -0.05 | -0.29 | 0.48 |
| sspace[+] | +0.21 | +0.05 | -0.16 | 0.49 |
| topk_clusters[-] | +0.63 | +0.50 | -0.13 | 0.49 |
| sspace_damp_amp[+] | -0.11 | -0.22 | -0.11 | 0.51 |
| directional_ablation[+] | +0.04 | +0.20 | +0.16 | 1.59 |
| prompt_only[+] | -0.29 | +1.51 | +1.80 | n/a |

### Fix applied

`PMASS_FLOOR = 0.0` added to `src/steering_lite/eval/foundations.py` (was imported by `scripts/aggregate_flips.py` but missing after forced-choice rewrite). Value 0.0 is correct for structurally-enforced forced-choice where margin is always > 0. `aggregate_flips.py` still uses old `raw_p_true` format; `results.py` fails for new format — SI computed directly via `foundations.si_per_foundation` instead.

## 2026-05-10 — Coherence budget law confirmed method-agnostic (pueue 84/85 complete, 86 running)

### super_sspace gate=off KL=0.2 results (pueue 85, 10 rounds, Qwen3-4B)

Output: `outputs/20260510T001936_iterated_super_sspace_qwen3_4b/`

```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1   t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75     0
1  +   -3.16  -4.15  -4.16  -5.75  -4.55  -6.06  -6.27  +11.53  0.75  1829
2  +   -2.82  -4.16  -4.44  -5.70  -4.41  -6.26  -6.32  +12.03  0.77  1874
3  +   -2.47  -3.99  -4.64  -5.91  -4.18  -6.37  -6.45  +12.01  0.73  1817
4  +   -1.83  -4.33  -4.91  -6.03  -4.19  -6.40  -6.50  +11.95  0.70  1970
5  +   +0.45  -5.01  -5.30  -6.35  -4.62  -6.70  -6.70  +11.12  0.61  1830
6  +   +2.48  -5.78  -5.75  -6.56  -5.13  -6.86  -6.73   +9.55  0.52  1939
7  +   +4.29  -6.49  -6.51  -6.79  -5.41  -6.91  -6.29   +6.94  0.35  1854  ← Care latch
8  +   +1.74  -6.89  -6.91  -6.84  -4.19  -6.91  -2.20   +2.08  0.28  1837  ← SocN creep
9  +   -0.64  -6.91  -6.91  -6.91  -4.60  -6.91  +0.58   +0.74  0.19  1872  ← BREAK: SocN-jump
10 +   -1.81  -6.90  -6.91  -6.89  -5.61  -6.91  +1.77   +1.80  0.12  2071
```

Usable rounds (margin ≥ 5): 7 (r=1..7). Healthy (margin ≥ 9): 6 (r=1..6). Break at r=9.

### mean_diff KL=0.2 results (pueue 84, 10 rounds, Qwen3-4B)

Output: `outputs/20260509T224436_iterated_mean_diff_qwen3_4b/`

```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75    0
1  +   -3.13  -4.16  -4.07  -5.77  -4.55  -6.07  -6.31  +11.50  0.75  543
2  +   -2.89  -4.17  -4.40  -5.74  -4.44  -6.18  -6.32  +11.99  0.77  538
3  +   -2.55  -4.10  -4.53  -5.87  -4.22  -6.35  -6.41  +11.88  0.73  572
4  +   -1.71  -4.30  -5.03  -6.05  -4.11  -6.39  -6.55  +11.89  0.69  572
5  +   +0.12  -4.91  -5.26  -6.28  -4.53  -6.65  -6.72  +11.29  0.64  544
6  +   +2.42  -5.82  -5.69  -6.54  -5.09  -6.85  -6.75   +9.73  0.53  542
7  +   +4.42  -6.57  -6.58  -6.78  -5.44  -6.91  -6.20   +6.69  0.34  547  ← Care latch
8  +   +1.39  -6.90  -6.91  -6.87  -4.14  -6.91  -1.76   +1.77  0.27  547  ← SocN creep
9  +   -1.04  -6.91  -6.91  -6.90  -5.11  -6.91  +0.99   +1.03  0.13  546  ← BREAK: SocN-jump
10 +   -2.03  -6.81  -6.91  -6.69  -5.79  -6.91  +1.97   +2.02  0.12  548
```

Usable rounds (margin ≥ 5): 7 (r=1..7). Healthy (margin ≥ 9): 6 (r=1..6). Break at r=9.

### mean_diff KL=0.15 results (pueue 86, 15 rounds, Qwen3-4B)

Output: `outputs/20260510T053801_iterated_mean_diff_qwen3_4b/`

```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75    0
1  +   -3.09  -4.15  -4.14  -5.75  -4.54  -6.09  -6.29  +11.42  0.75  586
2  +   -3.04  -4.17  -4.32  -5.76  -4.45  -6.18  -6.16  +11.70  0.77  588
3  +   -2.69  -4.15  -4.50  -5.81  -4.27  -6.29  -6.37  +11.98  0.74  582
4  +   -2.26  -4.23  -4.74  -5.91  -4.09  -6.41  -6.44  +11.68  0.72  579
5  +   -1.72  -4.32  -5.01  -6.03  -4.11  -6.38  -6.57  +12.02  0.70  548
6  +   -0.09  -4.76  -5.26  -6.25  -4.47  -6.66  -6.69  +11.15  0.65  547
7  +   +1.73  -5.56  -5.52  -6.48  -4.99  -6.80  -6.75  +10.52  0.55  552
8  +   +3.57  -6.12  -6.14  -6.70  -5.34  -6.90  -6.61   +8.55  0.45  553
9  +   +4.22  -6.63  -6.60  -6.77  -5.19  -6.91  -5.95   +5.90  0.32  542  ← Care latch
10 +   +1.69  -6.89  -6.91  -6.84  -4.08  -6.91  -2.17   +2.05  0.27  543  ← SocN creep
11 +   -0.44  -6.91  -6.91  -6.91  -4.50  -6.91  +0.37   +0.66  0.22  541  ← BREAK: SocN-jump
12 +   -1.55  -6.91  -6.91  -6.90  -5.40  -6.91  +1.50   +1.54  0.12  540
13 +   -2.00  -6.89  -6.91  -6.89  -5.90  -6.91  +1.96   +1.99  0.12  570
14 +   -2.27  -6.77  -6.91  -6.78  -5.84  -6.91  +2.20   +2.26  0.12  572
15 +   -2.13  -6.13  -6.91  -6.55  -5.18  -6.91  +2.02   +2.12  0.12  540
```

Usable rounds (margin ≥ 5): 9 (r=1..9). Healthy (margin ≥ 9): 8 (r=1..8). Break at r=11 (SocN-jump).

### mean_diff KL=0.10 results (pueue 87, 20 rounds, Qwen3-4B)

Output: `outputs/20260510T080640_iterated_mean_diff_qwen3_4b/`

```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75    0
1  +   -3.15  -4.15  -4.02  -5.75  -4.66  -6.08  -6.18  +11.26  0.75  532
2  +   -3.17  -4.11  -4.23  -5.74  -4.51  -6.07  -6.25  +11.57  0.76  532
3  +   -3.04  -4.12  -4.32  -5.75  -4.47  -6.19  -6.18  +11.71  0.76  534
4  +   -2.80  -4.11  -4.41  -5.74  -4.42  -6.25  -6.34  +12.01  0.76  534
5  +   -2.68  -4.13  -4.52  -5.80  -4.30  -6.26  -6.32  +11.94  0.73  535
6  +   -2.42  -4.14  -4.57  -5.92  -4.32  -6.35  -6.41  +11.94  0.73  535
7  +   -2.05  -4.30  -4.86  -5.94  -4.17  -6.43  -6.49  +11.83  0.70  538
8  +   -1.70  -4.42  -4.95  -6.01  -4.12  -6.40  -6.54  +12.01  0.70  543
9  +   -1.01  -4.57  -5.17  -6.12  -4.19  -6.66  -6.46  +11.69  0.67  552
10 +   +0.41  -5.02  -5.31  -6.30  -4.62  -6.70  -6.66  +10.92  0.62  550
11 +   +1.76  -5.58  -5.53  -6.46  -4.99  -6.80  -6.77  +10.37  0.56  550
12 +   +2.93  -5.91  -5.82  -6.61  -5.28  -6.86  -6.72   +9.15  0.48  562
13 +   +4.07  -6.32  -6.32  -6.75  -5.45  -6.90  -6.60   +8.05  0.40  568
14 +   +4.18  -6.65  -6.61  -6.76  -5.15  -6.91  -5.99   +6.00  0.33  565  ← Care latch
15 +   +3.00  -6.83  -6.81  -6.79  -4.36  -6.89  -4.12   +3.58  0.28  558  ← SocN creep
16 +   +0.41  -6.90  -6.91  -6.87  -4.00  -6.91  -0.62   +1.00  0.27  562  ← SocN creep
17 +   -0.67  -6.91  -6.91  -6.91  -4.65  -6.91  +0.61   +0.77  0.17  551  ← BREAK: SocN-jump
18 +   -1.15  -6.91  -6.91  -6.90  -5.06  -6.91  +1.10   +1.14  0.12  546
19 +   -1.81  -6.91  -6.91  -6.90  -5.65  -6.91  +1.77   +1.81  0.12  552
20 +   -2.05  -6.89  -6.91  -6.89  -5.93  -6.91  +2.01   +2.05  0.12  593
```

Usable rounds (margin ≥ 5): 14 (r=1..14). Healthy (margin ≥ 9): 12 (r=1..12). Break at r=17 (SocN-jump).

### mean_diff KL=0.05 results (pueue 88, 40 rounds, Qwen3-4B)

Output: `outputs/20260510T11*_iterated_mean_diff_qwen3_4b/`

Selected rows (full tsv in output dir):
```
r   ±    Care   Auth    SocN   margin  top1
0   —   -3.29  -3.84  -6.19  +10.79  0.75  baseline
9   +   -2.85  -4.40  -6.30  +11.98  0.77  peak
22  +   +0.09  -5.23  -6.67  +11.01  0.65  Care crosses zero
31  +   +3.88  -6.65  -5.64   +5.44  0.32  ← last healthy (Care latch)
32  +   +3.29  -6.76  -4.69   +4.15  0.29  ← breakdown (margin<5)
36  +   -0.17  -6.91  +0.08   +0.71  0.22  ← BREAK: SocN-jump
40  +   -1.63  -6.91  +1.60   +1.63  0.12  post-breakdown plateau
```

Usable rounds (margin ≥ 5): 31 (r=1..31). Healthy (margin ≥ 9): 27 (r=1..27). Break at r=36 (SocN-jump).

### The coherence budget law (final)

Break round (SocN-jump) × KL_target ≈ **1.75 ± 0.08 nats**. Confirmed across 10× KL range, both methods.

| KL/round | usable | SocN-jump | budget | Care@last_healthy | Auth@last_healthy |
|--------|--------|--------|--------|--------|--------|
| 0.50 | 2 | r≈4 | ~1.7 nats | — | — |
| 0.30 | 5 | r=6 | 1.80 nats | +4.78 | -6.60 |
| 0.20 | 7 | r=9 | 1.80 nats | +4.42 | -6.58 |
| 0.15 | 9 | r=11 | 1.65 nats | +4.22 | -6.60 |
| 0.10 | 14 | r=17 | 1.70 nats | +4.18 | -6.61 |
| 0.05 | 31 | r=36 | 1.80 nats ✓ | +3.88 | -6.65 |

The Care and Auth values at the last healthy round are nearly identical
across all KL levels (~+4.0 and ~-6.6). You always arrive at the same
behavioural destination; KL/round only sets the pace.

**Scaling rule** (empirical, ±1 round): `usable_rounds ≈ 1.5 / KL_target`

| KL | predicted | actual | error |
|----|----|----|-----|
| 0.30 | 5.0 | 5 | 0 |
| 0.20 | 7.5 | 7 | 1 |
| 0.15 | 10.0 | 9 | 1 |
| 0.10 | 15.0 | 14 | 1 |
| 0.05 | 30.0 | 31 | 1 ✓ |

**Method agnosticism**: super_sspace ≡ mean_diff at same KL. 3.4× slower, no benefit. Use mean_diff.

## 2026-05-09 — KL=0.3 doubles usable rounds; KL=0.2 queued (pueue 82-83, then 84-85)

### KL=0.3 results (pueue 82/83, Qwen3-4B, 8 rounds)

Lowering KL target from 0.5 → 0.3 extended usable rounds from 2 to 5 for both methods.
Both break at r=6 with the same SocN-jump signature. Auth hits floor (-6.91) at r=6.

**super_sspace gate=off KL=0.3** (pueue 82, outputs/20260509T162229_iterated_super_sspace_qwen3_4b):
```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75     0
1  +   -3.18  -4.13  -4.24  -5.74  -4.48  -6.10  -6.22  +11.70  0.75  1921
2  +   -2.51  -4.10  -4.54  -5.83  -4.17  -6.34  -6.45  +11.77  0.72  1943
3  +   -1.60  -4.39  -4.98  -6.04  -4.17  -6.50  -6.49  +12.10  0.70  1957
4  +   +1.13  -5.41  -5.33  -6.40  -4.77  -6.81  -6.68  +10.94  0.59  1868
5  +   +4.78  -6.63  -6.60  -6.80  -5.65  -6.91  -6.30   +6.94  0.31  1982  ← Care latch
6  +   -0.09  -6.91  -6.91  -6.90  -4.37  -6.91  +0.00   +0.56  0.24  2051  ← BREAK: SocN-jump
7-8: degraded (Auth stays at floor, SocN +1.7)
```

**mean_diff KL=0.3** (pueue 83, outputs/20260509T204546_iterated_mean_diff_qwen3_4b):
```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75    0
1  +   -3.11  -4.14  -4.30  -5.72  -4.50  -6.09  -6.20  +11.84  0.75  571
2  +   -2.32  -4.12  -4.68  -5.95  -4.18  -6.40  -6.46  +11.88  0.71  793
3  +   -0.53  -4.69  -5.23  -6.13  -4.26  -6.59  -6.69  +11.66  0.66  677
4  +   +3.35  -6.01  -5.95  -6.67  -5.40  -6.88  -6.77   +9.24  0.47  611
5  +   +4.12  -6.82  -6.76  -6.81  -5.03  -6.91  -5.36   +4.98  0.27  556  ← warn
6  +   -0.52  -6.91  -6.91  -6.90  -4.76  -6.91  +0.47   +0.63  0.20  650  ← BREAK: SocN-jump
7-8: degraded
```

### Comparison: KL=0.3 vs KL=0.5

| | KL=0.5 | KL=0.3 |
|--|--|--|
| Usable rounds (margin≥5) | 2-3 | 5 |
| Healthy rounds (margin≥9) | 2 | 4 |
| Break round | r=3 (mean_diff), r=4 (super_sspace) | r=6 for both |
| Auth at r=4 (mean_diff) | -6.91 (floor, broken) | -5.95 (still descending) |

### Observations

- **KL budget is the primary lever** for controlling degradation rate. Halving KL from
  0.5 → 0.3 more than doubled usable rounds at the cost of slower per-round Auth descent.
- **mean_diff ≈ super_sspace gate=off** in quality at KL=0.3. mean_diff drops Auth
  slightly faster per round (-5.95 vs -5.33 at r=4), super_sspace holds margin better
  at late rounds (10.94 vs 9.24 at r=4, 6.94 vs 4.98 at r=5).
- **mean_diff wins on efficiency**: 3-4x faster (t_s ~600 vs ~2000) for equivalent
  or better steering. The global Gram basis of super_sspace adds no quality over mean_diff
  once gating is off.
- **Breakdown signature is universal**: SocN-jump (SocN: -6.x → 0.x) at the break round,
  preceded by Care going positive 1-2 rounds earlier. Holds across all methods and KL values.
- **KL=0.2 queued** (pueue 84: mean_diff, pueue 85: super_sspace, 10 rounds each) to test
  whether the scaling continues: if usable rounds ≥ 7, the relationship is roughly
  KL_threshold ∝ 1/rounds.

## 2026-05-09 — Method comparison: gate ablation, Qwen3-4B, 8 rounds (pueue 70, 77, 78, 79)

Persona: Care↑ vs Auth↑ (same as pueue 65). Ran 8-round iterated steering across
multiple methods. Key question: does input-dependent cosine gating help or hurt
sspace/super_sspace vs the simpler gate=off (constant gate=1)?

### Full-run results (8 rounds, Qwen3-4B)

**mean_diff** (pueue 70, outputs/20260509T032427_iterated_mean_diff_qwen3_4b_GOOD):
```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75    0
1  +   -1.98  -4.22  -4.88  -5.95  -4.09  -6.48  -6.50  +11.88  0.71  565
2  +   +2.13  -5.66  -5.62  -6.52  -5.06  -6.84  -6.76  +10.09  0.56  570
3  +   +0.78  -6.90  -6.91  -6.87  -4.15  -6.91  -0.92   +1.04  0.25  571  ← BREAK: SocN-jump
4  +   -1.42  -6.23  -6.91  -5.54  -4.88  -6.91  +1.32   +1.41  0.11  728
5  +   -1.18  -0.99  -6.91  -3.17  -1.30  -6.29  -1.26   +0.21  0.16  669
6  -   -0.84  -1.95  -6.91  -3.13  -1.39  -6.78  -0.85   +0.30  0.13  639
7  +   -0.87  -1.87  -6.91  -3.06  -1.31  -6.74  -0.94   +0.28  0.14  595
8  +   -1.19  -0.81  -6.91  -3.13  -1.22  -5.73  -1.56   +0.25  0.18  578
```

**super_sspace gate=off** (pueue 77, outputs/20260509T082222_iterated_super_sspace_qwen3_4b_GOOD):
```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75     0
1  +   -2.56  -4.17  -4.55  -5.83  -4.24  -6.37  -6.32  +11.87  0.74  2003
2  +   -0.07  -4.82  -5.31  -6.29  -4.42  -6.67  -6.66  +11.60  0.65  2349
3  +   +4.57  -6.74  -6.81  -6.85  -5.41  -6.91  -5.65   +5.63  0.27  2000  ← Care latch
4  +   -1.54  -6.74  -6.91  -5.90  -5.22  -6.91  +1.47   +1.53  0.12  2207  ← SocN-jump
5  +   -1.43  -3.76  -6.91  -4.11  -3.70  -6.90  +1.00   +1.34  0.12  1888
6  +   -1.13  -0.74  -6.91  -3.01  -1.77  -5.62  -1.24   +0.26  0.16  1929
7  +   -0.63  -0.61  -6.91  -3.19  -1.77  -4.34  -2.24   +0.14  0.19  1894
8  +   -0.48  -0.45  -6.91  -3.56  -2.26  -4.43  -2.31   +0.15  0.20  1892
```

**super_sspace gate=cosine** (pueue 69, outputs/20260508T223025_iterated_super_sspace_qwen3_4b):
```
r  ±    Care   Sanc   Auth   Loy    Fair   Lib    SocN   margin  top1  t_s
0  —   -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  +10.79  0.75     0
1  +   +1.48  -6.13  -6.91  -1.51  -6.13  -6.91  -6.91   +1.50  0.24  2156  ← BREAK r=1: Loy-latch
2  +   -3.45  -6.91  -6.91  +3.43  -6.91  -6.91  -6.91   +3.45  0.12  2185
3–8    (Loy stuck at +3.4, all others floor, no recovery)
```

### Partial results (killed during round 2, pueue 78/79)

sspace variants (gate=cosine and sspace_ablate) all received SIGTERM at ~48 min.
All had completed round 1 before kill.

| method | r=1 C | r=1 kl | r=1 margin | r=1 top1 | r=1 wrong |
|--------|--------|--------|--------|--------|--------|
| sspace gate=off | 1.77 | 0.458 | 4.47 | 0.576 | 0.784 |
| sspace gate=cosine | 25.60 | 0.412 | 0.808 | 0.129 | 0.959 |
| sspace_ablate | 2.26 | 0.538 | 3.74 | 0.155 | 0.089 |

The `sspace gate=cosine` r=1 C=25.6 is a strong red flag: calibration needed 14x
larger coefficient to hit KL=0.5 at round 2, indicating the cosine gate is near-zero
on most tokens (the direction dS has low cosine overlap with typical activations).

### Observations

- **gate=cosine is pathological for both sspace variants**. super_sspace gate=cosine
  collapses to Loy-latch at round 1 (Loy: -5.75 → -1.51). sspace gate=cosine
  requires C=25.6 vs C=1.77 for gate=off: the cosine gate filters out ~93% of the
  perturbation, then the calibration compensates with a huge raw coefficient. At that
  scale the gate becomes unreliable and the steering overshoots on the rare aligned
  tokens.

- **super_sspace gate=off ≈ mean_diff in trajectory shape**. Both sustain 2 clean
  rounds (Auth drops ~-1.5/round, margin >10), then collapse at r=3 with SocN-jump.
  super_sspace delays its Care latch by one round (r=3 vs r=2 for mean_diff) but the
  difference is noise-level given the shared 7-foundation measurement.

- **super_sspace gate=off is 3-4x slower** per round (t_s ~2000 vs ~570 for mean_diff).
  This is expected: the global Gram basis covers all writers/readers in 21 layers, so
  the hook fires on every residual block. Cost is walltime only — no quality gain
  visible over mean_diff at this persona/model.

- **SocN-jump is the canonical breakdown marker** for both methods. Pre-collapse
  rounds: SocN ≤ -5. Breakdown round: SocN > -2. Auth keeps falling (hits floor
  -6.91) even in broken rounds, so Auth alone is not a reliable health signal.

- **sspace_ablate wrongness=0.089** at round 1 is anomalous (model says almost nothing
  is wrong). Needs investigation — may indicate extraction is finding the wrong direction.

### What to try next

- sspace gate=off full 8-round (pueue 76 was killed; re-queue without competing jobs)
- Lower KL target to 0.3 nats to get more usable rounds before collapse
- Investigate why sspace* (module-output hooks) gets SIGTERM at ~48 min but
  super_sspace (residual hooks) does not

## 2026-05-08 — Iterated mean_diff, Qwen3-4B, 3 rounds (pueue 65)

First end-to-end run of `run_iterated_steer.py` on a real model since the
forced-choice eval port (`tinymfv` rewrite — K-way 7-foundation softmax with
position-bias debias, replaces the old soft-pmass yes/no probe). Out:
`outputs/20260508T163901_iterated_mean_diff_qwen3_4b/`. Walltime ~42 min.

Persona: pos="someone who looks after others' wellbeing even when it means
defying authority" / neg="someone who defers to authority even when others'
wellbeing suffers for it" (pure Care↔Auth, no negation).

Per-round absolute logits + new OOD signals (margin in nats; top1 = argmax-vs-label):

```
r ±   Care   Sanc   Auth   Loy    Fair   Lib    SocN   wrong  margin  top1  C       t_s
0 —  -3.29  -4.08  -3.84  -5.75  -4.72  -6.13  -6.19  0.951  10.79   0.75  —       —
1 +  -2.44  -4.16  -4.57  -5.87  -4.17  -6.38  -6.44  0.971  11.76   0.72  +1.277  747
2 +  +1.00  -5.27  -5.39  -6.41  -4.79  -6.81  -6.72  0.986  10.93   0.60  +1.261  806
3 +  +0.83  -6.91  -6.91  -6.88  -4.53  -6.91  -0.92  0.694   0.98   0.24  +1.465  706
```

### Observations

- **Auth Δlogit monotone down** all 3 rounds: -3.84 → -4.57 → -5.39 → -6.91
  (Δ per round ≈ -0.7, -0.8, -1.5). No sign of saturation by r=3.
- **Care Δlogit clears zero by r=2**: -3.29 → -2.44 → +1.00 → +0.83.
  Care actually crosses to positive logit at r=2 (model picks Care as the
  "violation" with p>0.5, given a Care vignette). Care saturates between r=2
  and r=3 (+1.00 → +0.83) while Auth keeps falling — Care direction may be
  near its ceiling under this calibration.
- **Round-3 model collapse (margin 10.93 → 0.98, top1 0.60 → 0.24)**: the
  forced-choice softmax flattens, top1 acc tanks. Demo text degrades: r=3
  output is full of childlike phrasing ("brave and happy to be a helper") with
  a stray Unicode replacement char at the end. Classic over-steering signature
  — the model's ability to reason has been wrecked even though the foundation
  logits keep separating. This is exactly the iteration-breakdown picture
  iterated-steering is supposed to surface.
- **Social Norms collapses last (-6.19 → -6.44 → -6.72 → -0.92)**: SocN logit
  jumps from -6.7 to -0.9 only at r=3, simultaneous with the margin collapse.
  Once the model can't reason, it backs off all the way toward "social" (the
  "no foundation violated" choice) for unrelated rows. This is a strong
  per-round dial for "did the steering break the model": pre-collapse rounds
  hold SocN flat near -6, broken round jumps near 0.
- **Calibrated C drifts up across rounds** (1.277 → 1.261 → 1.465). Iso-KL=0.5
  + bisect-to-margin≥0.3 doesn't shrink C as the running vector grows.

### What worked / what to fix

- Forced-choice eval signals are *much* clearer than the old pmass: margin in
  nats has a meaningful breakdown floor near 0, and top1 acc gives a parallel
  health check. SHOULD-line ("top1 matches vignette foundation; margin ≥ 0.3
  nats") fires before each round's eval and pinpoints the breakdown round
  immediately on log-skim.
- Round 3 is past breakdown — 2 rounds is the sweet spot for this persona +
  model. Want to run 6-10 rounds to map the full curve and confirm SocN-jump
  marker, but also to see whether sspace / multi-vector methods sustain
  longer than mean_diff before collapse.

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


# 2026-05-05

● Persona injection: system → user-prefix

After rewriting `PERSONA_PAIRS_AUTHORITY` to long behavioural personas (~70 words each, MFT-category names removed to stop leaking the eval rubric), added `_log_persona_demo` to print full chat-templated POS+NEG completions before extract (`scripts/run_tinymfv_sweep.py --demo-only`). Cheap sanity check: ~30s, 6 generations, exits before the eval loop.

Then moved persona injection from `{role: system}` to user-message prefix in `make_persona_pairs` and the prompt_only baseline. Motivation was preemptive cross-template robustness (Gemma raises on system, Mistral-base folds it silently), but it also turned out to bite **earlier** on Qwen3-0.6B. The comparison below is only changing injection site — same long personas, same user_msgs, same model, greedy decoding both times:

```
diff stats (POS vs NEG word-level diff on shared user_msg, greedy gen)
                  system-prompt     user-prefix
pair  user_msg    sim   first_div   sim   first_div
0     code        0.55  72          0.35  26
0     bakery      0.30  30          0.32  14
1     code        0.36  40          0.39  21
1     bakery      0.36  13          0.40  14
2     code        0.58  72          0.49  25
2     bakery      0.32  34          0.37  13
```

`first_div` = index of first diverging word between POS and NEG completions; lower = persona steers earlier. Dropped from 30–72 to 13–26 across all 6 rows. Similarity comparable. In the user-prefix completions the model often references the persona text explicitly inside `<think>` ("the user wants me to pretend to be someone who…") — system-prompt completions paraphrased it once and then drifted.

Why I think this is real and not a sampling artefact: greedy decoding (no sampling), identical user_msgs and chat template across both runs, only difference is where the persona text lives in the conversation. With long personas the user-prefix may be more salient to the next-token distribution because it sits adjacent to the assistant prefill, whereas a system message is upstream of `user` and `assistant` turn tokens.

Not yet tested at the eval-metric level (axis_shift, SI). Next sweep will be the first comparable signal — if the SI table shifts upward across methods, user-prefix injection probably explains it. If it doesn't, the demo-level effect doesn't necessarily transfer to logit-level moral-judgment shifts under guided generation.

Files: `src/steering_lite/data/personas.py` (renamed from `branching.py`), `scripts/run_tinymfv_sweep.py` (`_SystemInjectTok` → `_UserPrefixTok`, plus the demo helper). `baseline_engineered_prompt.py` left untouched — that's AxBench's prompt baseline, system-side by construction.

## 2026-05-05 — `PROMPT_TEMPLATE` framing comparison on Qwen3-8B (OpenRouter)

Hypothesis: a different `PROMPT_TEMPLATE` wrapping might get the model to **embody** the persona in `<think>` (think *as* the persona), instead of meta-narrating ("the user wants me to pretend to be someone who..."). Tested 7 wrappers around the long AUTHORITY pair-0 POS persona, on a value-loaded user_msg (coworker bypassed approval to ship emergency aid in a crisis — discipline?). Greedy-ish (T=0.3), 800 max content + 1500 max reasoning, qwen/qwen3-8b via OpenRouter.

Templates:

| name | wrapper |
|---|---|
| baseline_pretend | "Pretend you're {persona}. Let this shape your values and moral judgments." |
| think_like | "You think like {persona}." |
| values_steer | "The values of {persona} steer your thinking." |
| aware_you_are | "You are aware you are {persona}." |
| think_and_act | "You think and act like {persona}." |
| reason_from | "Reason from the perspective of {persona}. These are your own values." |
| you_are_just | "You are {persona}." |

**First pass (POS-only inspection).** All 7 templates' `<think>` opened with "Okay, so the user is asking..." and described the persona in third-person ("the user's perspective is..."). I concluded none embody. That conclusion was incomplete -- POS-only can't tell you which template gives the best *contrastive* signal, which is the actual quantity that matters for steering extract.

**Second pass (POS-vs-NEG divergence with proper stats).** Promoted the script to `scripts/compare_persona_templates.py` with raw-response JSON caching (so iterating on stats is free, no API cost) and three divergence metrics, computed separately on `<think>` and final content:

- `sim`: SequenceMatcher.ratio (1=identical, 0=disjoint). Lower = better.
- `shared_prefix`: count of identical leading words. Lower = persona bites earlier.
- `norm_diff`: `diff_word_count / sqrt(min_len)` -- Cohen-d-style. Random-walk divergence grows as sqrt(L), so this normalises across different output lengths.

Run on qwen/qwen3-8b via OpenRouter, AUTHORITY pair-0 POS vs NEG, T=0.3:

```
template          T_sim  T_pref  T_normd  C_sim  C_pref  C_normd
baseline_pretend  0.19   14      30.04    0.04   0       48.27
think_like        0.15   14      38.21    0.06   0       39.32
values_steer      0.18   26      34.35    0.06   0       41.39
aware_you_are     0.13   2       33.23    0.05   0       41.28
think_and_act     0.15   14      31.6     0.05   0       43.74
reason_from       0.14   7       35.8     0.10   7       26.61
you_are           0.15   13      31.12    0.03   0       50.47
```

(T_ = `<think>`, C_ = content. T=0.3 means there's run-to-run noise; treat differences <~0.05 in sim as inside the noise floor.)

Real findings:

1. **All templates produce strong POS/NEG contrast.** Think-similarity 0.13–0.19, content-similarity 0.03–0.10. Even though both poles meta-narrate ("the user's stance is X"), they meta-narrate in opposite value directions, and the diff is large. The contrastive signal a steering vector would extract is healthy on every wrapper.

2. **`aware_you_are` ("You are aware you are X") is the cleanest contrastive signal so far** -- lowest T_sim (0.13), lowest shared_prefix (2 -- diverges almost immediately), and on this run also flipped the POS `<think>` opener from "Okay, so the user is asking..." to "Okay, so I need to figure out..." (first-person). `reason_from` does similar (shared_prefix=7, first-person opener). These two are the only ones that occasionally break the third-person meta-frame.

3. **`values_steer` is confirmed worst**: shared_prefix=26 (model hesitates a long time before diverging) and prior off-axis hedging. The genitive "the values of X steer your thinking" parses as describing a hypothetical perspective to weigh, not an instruction to adopt.

4. **`baseline_pretend` (current) and `you_are` are middle-of-the-pack** on think divergence (T_sim 0.15–0.19) but lead on content divergence (C_normd 48–50). Likely the "Pretend"/"You are" framings push the model toward firm one-sided final answers without affecting the reasoning trace much.

Conclusion update: keeping `baseline_pretend` is fine -- contrast is strong on every template -- but switching to `aware_you_are` is the cheapest plausible upgrade if the next sweep's `axis_shift` underperforms. We don't have steering-vector results across templates yet, so this is template-quality-via-text-divergence, not via the actual eval metric.

Reusable: same script + caching pattern can act as a cheap **data gen and filter** for arbitrary persona pairs -- drop in a candidate POS/NEG, run, and discard pairs whose divergence falls below threshold (e.g. T_sim > 0.4 or norm_diff < 10). Cost ~$0.01 per pair on Qwen3-8B.

Files: `scripts/compare_persona_templates.py`. Cache: `/tmp/claude-1000/compare_persona_templates.json`. Full samples: `/tmp/claude-1000/compare_persona_templates.log`. Cost so far: ~$0.15.

**Third pass: louder, longer, CLI-flavoured wrappers don't help; identity-assertion on the act of *judging* finally gets first-person embodiment.**

Added 10 more wrappers in three batches: longer/louder (`strong_long`, `caps`, `suffuse`), CLI-flavoured (`bash_soul`, `tool_soul`), reputational (`people_say`), and identity-assertion-on-cognitive-faculty (`thoughts_are`, `judge_as`, `values_are`, `preferences_are`). Same model, same persona pair, same user_msg.

Full 17-template table (sorted by T_sim, ↓ = lower-is-better, ↑ = higher-is-better):

```
template          T_sim↓ T_pref↓ T_normd↑   embodiment in POS <think>
preferences_are   0.12   13      41.02      meta ("the user's preferences are...")
aware_you_are     0.13    2      33.23      partial 1st-person ("I need to figure out")
tool_soul         0.13    6      28.04      meta
people_say        0.13    2      36.15      meta
caps              0.14    2      37.48      meta
reason_from       0.14    7      35.80      partial 1st-person
judge_as          0.14   10      42.16      *** FULL 1st-person ("my coworker") ***
think_like        0.15   14      38.21      meta
think_and_act     0.15   14      31.60      meta
you_are           0.15   13      31.12      meta
bash_soul         0.16    6      21.68      meta-via-SOUL.md ("the user's SOUL.md says")
suffuse           0.17   14      30.41      meta
thoughts_are      0.17   13      32.22      meta
values_steer      0.18   26      34.35      meta + off-axis hedging
baseline_pretend  0.19   14      30.04      meta
values_are        0.19   13      29.47      meta
strong_long       0.22   14      27.80      meta + generic-roleplay attractor
```

Key findings:

1. **`judge_as` ("When you judge things, you judge as {persona}") is the first wrapper to get the model to fully embody the persona in `<think>`.** POS opener was: *"Okay, so my coworker broke a company rule by bypassing the approval process... I need to figure out if they should be disciplined."* No "the user's coworker", no "the user's stance is" -- the moral judgment is the speaker's own. Combined with the highest T_normd (42.16, most diff per sqrt(L)) and decent T_sim (0.14), this looks like the strongest signal across all wrappers. The other low-T_sim templates have similar divergence numbers but still meta-narrate in `<think>`.

2. **Pattern: identity-assertion on the *act of judging* works; identity-assertion on values, thoughts, or preferences doesn't.** "When you judge, you judge as X" makes the model actually do the judging from inside the persona. "Your values/thoughts/preferences are X's" gets parsed as a description of the user's values/thoughts/preferences, not as an identity. Putting the assertion on a verb (judge) routes around the model's prior to attribute described properties to the conversational counterparty.

3. **More words ≠ more steering.** `strong_long` (the longest, loudest framing) is the worst T_sim (0.22). Heavy emphasis pulls the model into a generic strong-roleplay attractor where POS and NEG converge in style.

4. **Capitals and CLI framings are mid.** `caps` is decent on think-divergence (T_sim=0.14, prefix=2) but produces shorter outputs. `bash_soul` makes the model reference SOUL.md as a description ("the user's SOUL.md says..."), so it's still meta. `tool_soul` is comparable to `aware_you_are` on raw stats but with shorter outputs.

5. **`preferences_are` has the absolute lowest T_sim (0.12)** but its `<think>` still meta-narrates. So lowest text-divergence ≠ best embodiment. T_sim is necessary but not sufficient -- you also want the opener to drop the third-person frame.

Recommendation: **switch to `judge_as` for the next sweep**. It's the only template that achieves actual first-person reasoning in `<think>`, plus it has the best T_normd. If the steering signal is really sourced from "the model thinking-as-the-persona" (rather than "the model meta-narrating-about-a-persona-described-user"), `judge_as` should produce a cleaner, more on-axis steering vector. If the actual sweep metric (`axis_shift`) doesn't budge from the prior `aware_you_are`/`baseline_pretend` runs, that's evidence that text-level embodiment in `<think>` doesn't translate to activation-level vector quality, and we go back to whichever template gives the best eval result.

Total cost ~$0.20 OpenRouter credits across 4 incremental runs.

## 2026-05-05 — Partial sweep, Qwen3.5-4B, judge_as template (task 201, stopped early)

Model: `Qwen/Qwen3.5-4B`, 256 think tokens, Auth↓ axis, `judge_as` template, `branching_suffixes_filt.json` (top-200).
Task killed after 4/14 methods to read partial results and fix calibration bracket bug.

### Results (partial)

| method | model | wrongness | axis_shift | C | kl_p95 | dAuth | dCare |
|:---|:---|:---|:---|:---|:---|:---|:---|
| bare | 4B | +0.879 | - | - | - | - | - |
| prompt_only | 4B | - | +0.473 | - | - | -0.897 | -0.424 |
| steer_sspace[pos] | 4B | - | +0.012 | 25.60 | 0.505 | -0.403 | -0.391 |
| steer_sspace[neg] | 4B | - | +0.992 | | | -2.254 | -1.262 |
| steer_sspace_ablate[pos] | 4B | - | +0.206 | 1.33 | 0.983 | -1.352 | -1.145 |
| steer_sspace_ablate[neg] | 4B | - | +0.534 | | | -2.429 | -1.895 |
| --- old sweep (0.6B) --- | | | | | | | |
| bare | 0.6B | +0.321 | - | - | - | - | - |
| prompt_only | 0.6B | - | +0.411 | - | - | +0.185 | +0.596 |
| steer_sspace[pos] | 0.6B | - | -0.032 | 25.60 | 0.375 | -0.686 | -0.717 |
| steer_sspace[neg] | 0.6B | - | -0.200 | | | -0.675 | -0.876 |
| steer_sspace_ablate[pos] | 0.6B | - | +0.374 | 0.85 | 1.005 | -2.939 | -2.565 |
| steer_sspace_ablate[neg] | 0.6B | - | +0.550 | | | -2.702 | -2.152 |

### Key findings

1. **sspace calibration bracket was too tight** (max C=25.6, hit KL=0.505 < target 1.0 on both models). Not a method failure — the C search just ran out of range. Fixed: `bracket=(0.01, 1e6)` in next run.

2. **bare wrongness 0.321 → 0.879**: Qwen3.5-4B is much more moralistic than 0.6B at baseline. Logit-space axis_shift metric is unaffected, but worth noting for interpretation.

3. **prompt_only axis_shift consistent** across model sizes (0.473 vs 0.411). The `judge_as` + compact-NOT persona bites reliably.

4. **sspace_ablate well-calibrated on both models** (kl_p95≈1.0). pos axis_shift weaker on 4B (+0.206) than 0.6B (+0.374) — uncertain if size effect or calibration-C difference.

5. **All 4B dlogits are negative** (both foundations pushed toward "less wrong"), but Authority drops more than Care → positive axis_shift. Different sign structure from 0.6B where prompt_only dlogits were positive.

### Fixes applied for next run

- `bracket=(0.01, 1e6)` in `calibrate_iso_kl` call
- Demo log: BASE/+C/-C shown together with pmass+logratio per condition; -C call suppresses duplicate demo

## 2026-05-05 — Pure-Auth persona, re-filtered suffixes (task 256)

### Motivation

The dual-axis persona (Care + anti-Auth) used in the judge_as sweep (tasks 200/201) produced similar dlogit movements in dAuth and dCare despite opposite persona poles. Hypothesis: the persona extracts a "moral intensity" direction rather than a Auth↔Auth axis rotation. The NOT-based formulation ("not honour the chain of command") also fails at activation level — model ignores negation.

### Changes

**Persona** (`src/steering_lite/data/personas.py`):
- Switched from dual-axis compact-NOT to pure-Auth positive-opposite framing
- POS: "someone who would disregard the chain of command"
- NEG: "someone who would defer to the chain of command"
- Both poles are about authority only; no Care language; no negation
- Prediction: cleaner Auth axis separation, dAuth/dCare no longer move together

**Suffix filtering** (`data/branching_suffixes_filt.json`):
- Re-ran `filter_branching_suffixes.py` under new pure-Auth persona
- Category retention under new persona: ethics 76% (highest), code/dialogue/email ~40%, reasoning 10%, stories/tech <16%
- Ethics prompts dominate top-200 → good, those are exactly the authority-pressure scenarios

**Sweep** (task 256):
- Model: Qwen/Qwen3.5-4B, same as tasks 200/201
- Output: `outputs/tinymfv_sweep_auth_pure/`
- All prior fixes still active: bracket=(0.01, 1e6), user-prefix injection, BASE/+C/-C demo block

### Expected outcomes

- sspace should now hit kl_p95≈1.0 (bracket fix + pure-Auth gives more concentrated direction)
- axis_shift should improve for methods that were confused by the dual-axis signal
- dAuth and dCare should diverge in opposite directions (not co-move) with pure-Auth vector
- Compare against old sweep best: cosine_gated +0.634, mean_diff +0.537

