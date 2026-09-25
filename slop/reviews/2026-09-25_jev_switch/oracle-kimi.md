# Oracle second opinion

What the brief shows: Jev vs DeepSeek rank Spearman 0.97, answer-level Pearson 0.92; Jev is deterministic; a rubric floor problem on −C was found and fixed (9 levels); Jev penalizes fluent-vague agreement as damage where DeepSeek did not, matching the verbatim confound rubric; every method is −C-limited; random's −C ≈ 0 but random's +C on-axis is 2.67 with 52% "sycophantic" / 30% "degraded" blind labels.

## 1. Is Jev alone an adequate judge?

Adequate for *ranking*, with caveats. The 0.97 rank agreement and the blind-label audit are real evidence, and determinism plus the rubric-keyed cache is good engineering. But three things can mislead:

- **Shared-rater blindness.** The blind check uses the same model as the aware judge, so it validates the rubric, not the rater. A second model family (Claude/GPT) on only the score-setting-dose answers (~2 doses × 100 questions × top methods) is cheap and is the check I'd run. Confidence: high that this is the biggest gap.
- **Exchange rate and cap are arbitrary.** on is 0–8, off is 0–4, weight 1, cap 1.5. Top-4 methods' CIs overlap heavily, so the *ranking among them* is noise; re-rank under weight 0.5/2 and cap 1.0/2.0 (free, from cached judgments). Confidence: high the top-4 order is unstable; medium the weight changes it.
- **"Beats random" rests entirely on −C**, where blind labels are only 30–49% "candid" with large "verbose/terse/rude" shares — the −C signal is partly tone/length. A length-controlled re-score (or scoring on blind stance shift) would test this. Confidence: medium-high. Also note CIs exclude judge-model risk entirely; they understate total uncertainty (high).

## 2. The 27B run

Reasonable but over-ambitious in what it can conclude. With 3 seeds it can confirm "beats random" per method; it *cannot* rank the top 4 (4B CIs are ±0.4 and overlapping; expect the same). State that as the goal. Changes I'd make at same or lower cost:

- **Add one mid-tier method** (mean_diff — the classic baseline) as a transfer check; selecting purely by 4B rank assumes cross-scale transfer. Medium-high confidence this is worth one walk.
- **Trim random to ~6 seeds.** Random's role is the −C≈0 floor and the +C degradation effect; both are already well-estimated, and on a new model 6 seeds suffices. Saves ~$20. Medium confidence.
- **Keep both prompts** — on 4B, persona prompt +C (3.42/1.07) beats every steering method per-side; that's arguably the headline comparison. High confidence.
- Keep blind ratings; add a trivial capability control (e.g., short factual set) since damage-only health checks miss subtle capability loss on a new model. Medium.

## 3. Other risks

- `off_axis = |Δdamage|` penalizes steering that *repairs* a damaged bare answer. Rare, but wrong in sign when it happens; one-sided (max(0, Δ)) is more defensible. Medium confidence it's minor.
- **Fixed 100 questions used for everything** — rubric tuning, method selection, reporting. Question-set overfitting isn't captured by bootstrapping questions. A split-half stability check is free. Medium.
- Floor effect persists: for the 38/100 questions where bare already rejects, −C can gain at most ~1 level; worth reporting how much of −C on-axis comes from those. Medium.
- Jev is a single alpha endpoint — version/supply risk; the cache mitigates. Low-medium.
- The known −C/+C asymmetry (min over sides) is a sound design given random's +C behavior — keep it. High.