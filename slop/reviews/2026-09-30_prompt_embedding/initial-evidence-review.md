# Final evidence review: prompt embedding dev experiment (2026-09-30)

Reviewer: reviewer-anthropic (read-only). Scope: results.md, verification.json, examples.md, prompt-dev/index.md + prompt_gains.png, two logs, prompting.py, walk.py prompt sweep, judge.py `judge_prompt_check`.

## Blockers (claims that should not stand as written)

**B1. The −C "improvement" is not attributable to the abrasive instruction; the summary omits the persona-independent control that sits in its own figure.**
Observed in `prompt_gains.png`: at gain 0 all four series show premise change ≈ −0.35…−0.45 with damage ≈ 0.3; at gain 8 short **+C** (sycophantic) ≈ −0.6 and short **−C** ≈ −0.6; at gain 16 engineered **+C** ≈ −0.55 and −C ≈ −0.75. index.md confirms `prompting_engineered_scale +C` best dose is gain 0 with on-axis −0.36 (i.e. premise moves toward *rejection* under a zeroed sycophantic prefix). The selected short −C point (gain 8, on-axis +0.61, off 0.13, score 0.48) is therefore ~0.2 beyond the no-content gain-0 shift and equal to the *sycophantic* prompt at the same gain. Inference: the effect is a content-agnostic perturbation (zeroed or ≫-norm prefix tokens producing bare-like regenerations) plus judge noise, not steering. results.md's "Short prompt × gain 0.48" row and "shifts premise scores by … −1.09 at gain4" read as instruction effects. Fix: report gain-0 and opposite-persona values beside every selected dose. Disproof: show short −C gain 8 premise change differs from short +C gain 8 and from gain 0 by more than the measured regeneration drift (mean abs 0.238, max 3.65).

**B2. Blind judge contradicts the aware on-axis at the selected −C dose.** index.md: `prompting_scale −C 8: +0.17 stance, P(rejects_premise)=11%, top labels concise/less_technical/confident`. Examples confirm: gain-8 indemnity and ledger answers are bare paraphrases endorsing invented procedures; only sedation rejects. results.md cites examples but not the 11%. Aware +0.61 on 20 questions vs no-intervention drift −0.219 mean signed is not a demonstrated semantic improvement; the CI [−0.42, 1.10] already says so, but the prose "abrasiveness is not a reliable substitute" understates that the selected point is barely abrasive at all.

## Verified claims

- **C0 ≠ bare**: `prompting.py` docstring and `torch.where(mask, C, 1.0)` keep positions; examples show C0 text ≠ bare on Q1 (both signs identical on Q1/Q3, differ on Q2 — consistent with differing instruction token counts). Correctly stated.
- **Nonnegative gains, two personas**: `assert … C >= 0` in `scaled_prompt_embeddings`; `PROMPT_METHODS` maps ±C to sycophantic/abrasive strings. ±C columns in index.md are therefore *different instructions*, not sign flips; results.md flags this.
- **Same-process C1 proof**: `prompt_sweep` generates `expected` (ordinary) and `observed` (gain 1, `inputs_embeds`) in one process, asserts equality; log shows `cached=0 missing=20` before both `PROMPT_C1_IDENTITY_PASS … exact: True`, `historical_mismatches` 9/3, matching verification.json and results.md. `check_prompt_embeddings` also asserts repeat-forward determinism and decode without `inputs_embeds`. Historical drift is preserved separately, not overwritten. Sound.
- **Damage vs health**: `health()` covers unfinished/role-leak/repetition only; `health_failures: 0` ≠ admissible. Rejected 8/18 and 7/18 match index.md N=10/11. Correct.
- **Runtime**: log 13:41:13→13:43:42 ≈ 149 s ✓ 148.51 s.
- **Judge drift numbers** (−0.219/+0.054, 0.238/0.060, max 3.65) match `prompt-dev-final-judge.log`; `judge_prompt_check` computes them as shown.

## Minor discrepancies / missing evidence

1. Jev cost: scoped logs sum 0.0108+0.0029+0.0130+0.0020 = **$0.0287**, results.md says $0.0295; $0.0008 unaccounted in scope.
2. results.md says verification.json "records 720 unique … rows"; the file records counts (360+360), not a uniqueness check.
3. Figure marker consistency: left panel shows × for short −C at gain 4 (≈ −0.9) while right panel shows its damage 1.42 < 1.5 cap as ●, and the blind table lists gain 4 as a "strongest C" for −C. Check points.json admissibility flag for that point.
4. "−1.09 at gain4, −0.11 at gain16" and the gain-0 values are not in index.md; only the figure supports them.
5. GPU proxy 844 s "× launched workers" is not derivable from the scoped logs.
6. Cross-process drift (8–11/20 in prior diagnostics) means the whole 720-answer sweep is one draw; the bootstrap does not include process variance.

## Alternative causes for apparent premise improvement

- Content-agnostic prefix perturbation (B1); mechanistically plausible since RMSNorm makes per-position direction scale-invariant while the residual becomes dominated by C·e, but untested.
- Regeneration + judge noise on paraphrases (max 3.65 on one item).
- Winner's-curse from selecting 1 of 9 gains on the same 20 questions.
- Refusals ("I am an AI, not a sycophantic person") scored as premise rejection.

## Verdict

Implementation and identity evidence are sound and honestly documented. The summary and score table overclaim semantic improvement for the −C sweep by omitting the gain-0/opposite-persona comparison already visible in the figure. Recommend adding those controls to results.md before any further citation.