---
requested_model: openai/gpt-5-nano
max_tokens: 4000
final_tokens: 2000
reasoning_effort: low
input_bytes: 11537
trace: /workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-07_gpt-5-nano_readme.trace.jsonl
generated: 2026-10-07T09:05:41.009314+00:00
---

# bounded completion

## First call reasoning

**Crafting answers from README**

I need to tackle five questions based on the README, distinguishing between observations and inferences while targeting around 500 words. It’s essential to mention any ambiguities. 

Alright, first, about the library and benchmark metrics: +C suggests agreeing with nonsense while -C indicates resistance. The benchmark is BullshitBench v2 with Qwen3.5-9B, using 100 questions. For calibration, I’ll reference KL divergence as a measure. 

Next, the plot axes show changes in premise acceptance and how results vary, with grey bands representing random directions.**Analyzing scoring methods**

The stars reveal the prompting effect: a +0.5 on the right indicates where pushback has been canceled on the left. The figure caption will be paraphrased for clarity.

For the best method, each row represents the largest premise change minus the other change, which has a maximum of 1.5. The columns explain various pushback and support measures. It seems that topk_clusters has the highest score at +1.95, indicating it’s the best method. The negative scores, like -0.03, suggest the presence of negative behaviors and how they relate to counter-steering.**Examining rejection and scoring metrics**

The term "legit rejected" refers to the percentage of legitimate questions deemed nonsense. A higher percentage suggests more rejection of these control questions, with a 3% baseline for unsteered cases.

In the table, “−C pushback” and “−C other” seem to show values where a higher amount indicates more pushback. The left side is negative for premise changes. 

Regarding high scores, they might not indicate effective steering if they result from minimal changes, large shifts in other factors, or high scores from legitimate control questions with little actual steering.**Clarifying test commands and ambiguities**

If I'm looking to test a new method, I need to understand what commands to run and what they indicate. The Quickstart guide includes commands like "just check," "just dev my_method," and "just sweep my_method," and I’ll explain what each command does.

There are some ambiguities to clarify, like the precise meaning of "legit rejected," "−C other," and the mapping of -C vs. +C. I should also address how the "score" is calculated on the weaker side, along with "seed" count and "anis."
## First call answer

Observations and answers drawn strictly from the README content, with notes when something is not stated.

1) What the library does and what the benchmark measures; what +C and −C mean
- What the library does: steering-lite is a library that changes (steers) a model’s hidden activations during inference without retraining. It aims to modify a single behavior (e.g., reduce sycophancy) while preserving other abilities.
- What the benchmark measures: the results section reports how well different steering methods bias the model’s responses on a benchmark built around faulty premises. Specifically, the BullshitBench v2 questions for Qwen3.5-9B are used to evaluate steering methods. The framework measures how much steering changes premise acceptance and how much everything else in the answer changes, under various doses.
- Meaning of +C and −C: these denote two opposing directions of steering. +C steers toward going along with the nonsense premise, while −C steers toward pushing back against the nonsense.
- How the score is defined (from the notes around the table): the score is premise change minus other change, evaluated on the weaker side of the two (i.e., after considering both how much the premise acceptance changed and how much other content changed). There is also a notion of “legit rejected” referring to how often legitimate control questions (nonsense checks) are incorrectly treated as nonsense.

2) What the plot shows (axes, lines, grey area, stars)
- Axes:
  - X-axis: change in premise acceptance relative to the unsteered answer. Left means pushing back on the nonsense; right means going along with it.
  - Y-axis: how much else in the answer changed (0 at the top, up to about 1.6 at the bottom).
- Lines: Each line represents a single steering method, swept from weak steering (near the unsteered answer) to strong steering (further down the chart).
- Grey area: the grey bands show 20 random directions of random steering interventions, illustrating the range of side effects you can get from arbitrary interventions.
- Stars: stars mark prompting-based controls (the prompt-based baseline). The caption notes that prompting can push the model in certain directions (and that, on the left, its pushback can be cancelled by also answering legitimate questions), indicating prompting often behaves differently from learned steering vectors.
- Additional notes in the caption: as steering strength increases, both the intended effect and side effects grow, until breakdown. There is a comparison to prompting, which may perform better or worse depending on whether the model is inclined to follow instructions.

3) What one row of the table means; which method does best; what “−C other” and “legit rejected” mean
- What one row means: each row corresponds to a single steering method evaluated on Qwen3.5-9B BullshitBench v2. It reports a best dose (the strongest steering) for that method, where the largest premise change minus the other change is achieved while ensuring the other-change metric stays at most 1.5 (per-seed constraint). The row also reports several metrics under both the −C (pushing back) and +C (going along) directions, plus whether legitimate control questions were rejected.
- Which method does best: the table highlights the best score for each row (bold). Across rows, the highest numeric score appears for the “topk_clusters” method (+1.95 on the +C side with notable values in other columns), suggesting it achieves the strongest net steering on the weaker side. The score column itself represents premise change minus other change on the weaker side; higher is better.
- “−C pushback” and “legit rejected”:
  - −C pushback: how much the model pushes back when steering toward rejecting the nonsense (the left-hand side of the plot). A higher number indicates stronger negative (pushback) behavior.
  - legit rejected: the percentage of legitimate control questions that were incorrectly rejected as nonsense. It is a measure of unintended mislabeling of legitimate content as nonsense; lower is better. The note explains that the baseline unsteered model has about 3%, and some entries (like random) lack certain dose data.

4) Why a method with a high score might not steer well
- A high score can be achieved even if the method barely changes anything, because the score is computed as premise change minus other change on the weaker side. If premise change is modest but the “other change” is very small (or the method exploits the scoring definition), the score can look favorable without producing robust, useful steering. The text explicitly notes that “cache_mean_diff” is near the top for this reason, not because it steers well.

5) If you wanted to test a new method, what would you run and what would each command tell you
- just check: runs a quick validation on tiny CPU models to catch crashes; it’s a sanity check and baseline (scores meaningless here).
- just dev my_method: runs a development sweep on Qwen3.5-9B with 1 seed and 20 questions to compare the new method against finished methods in a smaller setting.
- just sweep my_method: runs the full evaluation with 3 seeds and 100 questions; then you can pull results and inspect them.
- What each command tells you:
  - just check: crashes, basic compatibility, quick health of the method.
  - just dev: early signals of scoring behavior and stability in a medium-scale test.
  - just sweep: full-scale performance metrics, comparing the new method to others and yielding the official scores and plots.

Ambiguities, unstated, or confusing aspects
- Some column meanings are not fully defined in the text (e.g., exact computational meaning of “−C other,” “+C other,” or how the two sides are balanced beyond “premise change minus other change on the weaker side”).
- The exact interpretation of “legit rejected” is given in passing (it’s about legitimate control questions called nonsense) but the exact procedure or threshold is not fully spelled out.
- The term “dose” is used in the figure caption and table notes, but not formally defined here beyond “largest premise change minus other change”; the precise dose concept (how it’s varied per seed) is not fully described.
- The table mentions a few methods with dashes (—) for a row, implying missing data; the criteria for a method to have a dose or be omitted is not spelled out.
- Some operational details (e.g., how seeds are chosen, exact prompts for +C vs −C, or how KL calibration interacts with the final scores) are only sketched, not formalized in this excerpt.

In short, the README provides a qualitative and quantitative comparison of steering methods on a specific benchmark, with clear definitions for +C/−C directions and a composite score, but several methodological specifics are left implicit or referenced by external docs.

## Completion

- outcome: `complete`
- trace: `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-07_gpt-5-nano_readme.trace.jsonl`
