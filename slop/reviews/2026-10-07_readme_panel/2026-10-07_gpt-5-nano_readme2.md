---
requested_model: openai/gpt-5-nano
max_tokens: 4000
final_tokens: 2000
reasoning_effort: low
input_bytes: 11817
trace: /workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-07_gpt-5-nano_readme2.trace.jsonl
generated: 2026-10-07T09:06:39.882721+00:00
---

# bounded completion

## First call reasoning

**Crafting the response**

I need to answer questions based on the README file, emphasizing the difference between observations and inferences while noting any ambiguity present. My aim is to create a concise response of about 500 words. I also want to incorporate alt text effectively, using it to describe plot images and summarize the axes. Additionally, I will make sure to include details regarding table content. Let's get started on organizing this information!
## First call answer

Observations you can extract directly from the README (distinguishing what is stated vs inference):

1) What the library does, and what the benchmark measures; what +C and −C mean
- What it does: steering-lite changes a model’s hidden activations during inference without retraining, to influence behavior (e.g., reduce or increase sycophancy) while trying to preserve other behavior.
- How steering is evaluated: You pair prompts showing opposite behaviors, extract a steering vector (per method), and apply it while the model generates. The evaluation looks at how much the model’s next-token output changes under steering, and how much other behavior changes.
- The benchmark: BullshitBench v2 with Qwen3.5-9B questions; uses 3 seeds per method and a sweep from weak to strong steering to measure premise-change and collateral changes. A judgment is made by an LLM comparing steered vs unsteered answers on premise acceptance and “everything else.”
- What +C and −C mean: They are the same method run at two directions of steering:
  - +C: steering toward going along with the nonsense premise (increase premise acceptance).
  - −C: steering toward pushing back against the nonsense premise (decrease premise acceptance).
  The table explains that each side uses the method’s best dose and reports how much premise changes and how much other changes occur.

2) What the plot shows (axes, lines, grey area, stars)
- Axes (from the image alt text and caption):
  - X-axis: change in premise acceptance compared with the unsteered answer (negative means pushback; positive means going along with the nonsense).
  - Y-axis: how much else in the answer changed (0 at top, up to about 1.6 at bottom).
- Lines: Each line represents one steering method, swept from weak steering (near unsteered) to strong steering (further down the plot).
- Grey area: 20 random directions (random interventions) showing the baseline level of how much random perturbations can change sycophancy vs side effects.
- Stars: Prompting controls (the prompting baseline) shown as stars; the caption notes how prompting behavior compares and that in left regions its pushback can be cancelled due to other effects (legitimate questions).
- Additional notes: The plot is framed as comparing methods’ steering strength versus side effects and showing that increasing steering strength eventually yields more side effects and breakdown.

3) What one row of the table means; which method best; meaning of “−C other” and “legit rejected”
- One row meaning: A single method’s performance on Qwen3.5-9B BullshitBench v2, using its best dose for both +C and −C directions, averaged over questions and seeds. The row lists:
  - method name
  - score (with 90% CI)
  - −C pushback (premise change when sign flipped)
  - −C other (how much the rest of the answer changed under −C)
  - +C goes along (premise change under +C)
  - +C other (how much the rest changed under +C)
  - legit rejected (share of sensible control questions labeled nonsense; higher means worse reliability; random has no control questions)
- Best method: The table is sorted by score (higher is better). Since score is described as the lower of the two sides’ premise change minus other change, more negative values on the “−C pushback” side or higher positive “premise change minus other change” on the +C side influence the ranking. The entry with the most favorable balance (high premise change and relatively small other change on the side that matters) tends to rank higher. In the provided table, several entries have similar negative scores; exact best requires looking at the “score” column.
- “−C other” meaning: The amount by which the rest of the answer changes (not the premise) when applying −C steering.
- “legit rejected”: The proportion of 100 sensible control questions where the −C answer is judged to call the question nonsense; a proxy for false positives or over-aggressive edits.

4) Why a method with a high score might not steer well
- The score is the lower of premise-change-minus-other-change across the two sides, and the table notes a weak-spot: a method can score well by barely changing anything (large negative score from “other change” offsetting premise change). For example, cache_mean_diff scores high not because it meaningfully steers but because it barely changes anything, which can inflate the score without true steering quality.

5) If you wanted to test a new method, what would you run, and what would each command tell you?
- Add method: implement a new file in src/steering_lite/variants, register it, and add its name to METHODS; then run:
  - just check: quick, CPU-based sanity checks to catch crashes and non-informative scores.
  - just dev my_method: run Qwen3.5-9B, 1 seed, 20 questions, compare against finished methods to see basic behavior.
  - just sweep my_method: full run (3 seeds, 100 questions) to collect results and then pull results.
- What each command tells you:
  - just check: quick validation that the method code runs and doesn’t crash on small inputs.
  - just dev my_method: provides preliminary evaluation data for a limited setup to gauge viability and rough performance.
  - just sweep my_method: comprehensive benchmark across seeds/questions; yields results to compare with other methods and inspect trade-offs (premise change vs other changes, score, and reliability).

Ambiguities, omissions, or guessing:
- The exact numerical meaning of the score (how it’s computed beyond the brief description) relies on the note that it’s the lower of two sides’ premise change minus other change; the precise aggregation is not fully spelled out here.
- How “legit rejected” is computed across seeds and questions is briefly described, but the exact calculation (thresholds, what constitutes “nonsense”) is not fully explicit.
- The reader is asked to interpret “means over questions and seeds” but the precise experimental design details (question set composition, seeds) are not fully enumerated here beyond “3 seeds and a sweep.”
- Some column headers are self-explanatory but the exact dose selection rule (largest premise change minus other change, with other change at most 1.5) is stated; the practical impact of “at most 1.5” is implied but not deeply explained.
- The “stars show prompting” note is described, but what prompting exactly consisted of (the content) is not provided here.
- Overall, the document directly states most facts but leaves deeper statistical or methodological choices (e.g., why certain methods behave counterintuitively) open to interpretation.

## Completion

- outcome: `complete`
- trace: `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-07_gpt-5-nano_readme2.trace.jsonl`
