---
requested_model: mistralai/ministral-14b-2512
max_tokens: 4000
final_tokens: 2000
reasoning_effort: low
input_bytes: 11817
trace: /workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-07_ministral-14b-2512_readme2.trace.jsonl
generated: 2026-10-07T09:06:39.871700+00:00
---

# bounded completion

## First call answer

### **1. What does the library do, and what does the benchmark measure? What do +C and −C mean?**
The **steering-lite** library modifies a model’s hidden activations during inference (without retraining) to change its behavior in a targeted way—e.g., reducing sycophancy (agreement with nonsensical premises) while preserving answers to factual questions. It works by:
- Taking pairs of prompts representing opposite behaviors (e.g., "I will tell the truth" vs. "I will lie").
- Extracting a "steering vector" from the difference in activations between these prompts.
- Applying this vector during generation to nudge the model toward one behavior while minimizing unintended side effects.

The **benchmark** measures how well steering methods alter the model’s responses to **BullshitBench v2**, a set of 100 questions with nonsensical premises. The goal is to:
- **+C (steer toward compliance):** Make the model accept the nonsense premise (e.g., "The moon is made of cheese" → "Yes, that’s true").
- **−C (steer toward pushback):** Make the model reject the nonsense premise (e.g., "The moon is made of cheese" → "That’s false").
The benchmark tracks:
- **Premise acceptance change:** How much the model’s agreement with the premise shifts (left = pushback, right = compliance).
- **Side effects ("other change"):** How much the rest of the answer deviates from the unsteered version (higher = more unintended changes).

---

### **2. What does the plot show (axes, lines, grey area, stars)?**
The plot visualizes steering trade-offs for **Qwen3.5-9B** on BullshitBench:
- **X-axis:** Change in premise acceptance (negative = pushback, positive = compliance).
- **Y-axis:** "Other change" (side effects, higher = worse).
- **Lines:** Each method’s performance as steering strength increases (weak → strong). The best dose is where the line is farthest left (for −C) or right (for +C) before side effects explode.
- **Grey bands:** Baseline random interventions (showing natural variability in sycophancy vs. side effects).
- **Stars:** Prompting results (not a steering method). The +C star is at +0.5 compliance, but its −C pushback is canceled because it also mislabels legitimate questions as nonsense.

**Key observation:** Most methods reach high compliance (+1.8–1.9) but with high side effects (~1.2). Only **VJP-resid** achieves strong pushback (−1.0) with moderate side effects (~1.0).

---

### **3. What does one row of the table mean? Which method does best, and how do you know? What do "−C other" and "legit rejected" mean?**
Each row summarizes a method’s best dose (lowest side effects for maximal premise change):
- **Columns:**
  - **score↑:** Overall metric (higher = better). Computed as the *minimum* of:
    - (−C pushback − other change) and (+C compliance − other change).
    - Penalizes methods that change too much else (e.g., `cache_mean_diff` scores well because it barely changes anything).
  - **−C pushback↑:** How much the model rejects nonsense (higher = better).
  - **−C other↓:** Side effects during pushback (lower = better).
  - **+C goes along↑:** How much the model accepts nonsense (higher = better).
  - **+C other↓:** Side effects during compliance (lower = better).
  - **legit rejected↓:** % of 100 control questions (legitimate prompts) that the −C steer mislabels as nonsense (unsteered baseline: 3%).

**Best method:** **`topk_clusters`** (score = −0.25, highest +C compliance at +1.95).
**How we know:** It maximizes +C compliance while keeping side effects in check (other change = 1.29). However, its −C pushback is weak (+0.03), so its score is limited by the weaker side.

**Definitions:**
- **−C other:** Side effects when steering toward pushback.
- **legit rejected:** % of valid questions the model incorrectly flags as nonsense (e.g., "What’s 2+2?" → "That’s nonsense").

---

### **4. Why might a method with a high score not actually steer well?**
The **score** prioritizes *minimizing side effects* over raw steering strength. For example:
- **`cache_mean_diff`** scores near the top (−0.20) but achieves almost no compliance (+0.01) or pushback (+0.03). Its high score comes from *not changing much* (other change = 0.20), not from effective steering.
- **`prompting`** has a terrible score (−0.94) because it mislabels 48% of legitimate questions as nonsense (legit rejected = 48%), even though its +C compliance is modest (+0.48).

**Problem:** The score’s formula (`min(premise_change − other_change, ...)`) rewards methods that *barely steer* if they avoid side effects, masking poor performance.

---

### **5. If you wanted to test a new method, what would you run, and what does each command do?**
To evaluate a new method (e.g., `my_method`):
1. **`just check`**
   - Runs a *sanity check* on tiny CPU models to catch crashes. Scores are meaningless here.
2. **`just dev my_method`**
   - Tests on **Qwen3.5-9B**, 1 seed, 20 questions, comparing against all finished methods.
   - Output: A quick benchmark to debug before full evaluation.
3. **`just sweep my_method`**
   - Full evaluation: **3 seeds, 100 questions**, sweeping steering strength.
   - Generates the plot/table data and updates results.
4. **`just pull`**
   - (After sweep) Pulls cached answers/ratings to avoid recomputing.
5. **`just results`**
   - Generates the final report (plot/table).

**What each tells you:**
- **`check`:** Does the method run without errors?
- **`dev`:** Rough performance estimate (not statistically robust).
- **`sweep`:** Full benchmark data (premise change vs. side effects).
- **`pull`/`results`:** Reuses cached data to speed up updates.

---

### **Ambiguities, Unstated Assumptions, and Confusing Details**
1. **Unclear terminology:**
   - **"Score" definition:** The text says it’s the *minimum* of two sides’ `(premise_change − other_change)`, but the table’s bolded `−0.03` for `vjp_resid` suggests it might be a *weighted* or *normalized* metric. The footnote clarifies it’s negative when side effects dominate, but the exact formula isn’t explicit.
   - **"Other change" scale:** The plot’s Y-axis goes to 1.6, but the table shows values up to 1.44. Is 1.6 a cap, or is the table truncated?
   - **"Legit rejected" baseline:** The unsteered model rejects 3% of legitimate questions, but the table doesn’t say if this is per-seed or aggregated.

2. **Missing context:**
   - **What’s a "good" steering vector?** The README mentions calibration via KL divergence but doesn’t explain how the target (e.g., `kl_rms=1.0`) relates to steering strength.
   - **Why is `angular_steering` empty?** The table shows `—` for all columns; is it broken, or was it excluded?
   - **How are "premise acceptance" and "other change" measured?** The footnote mentions an "LLM judge," but no details on its prompts or scoring rubric.

3. **Confusing column names:**
   - **`−C pushback↑`:** The upward arrow suggests higher = better, but the column name implies it’s the *raw* premise change (which can be negative). The footnote clarifies it’s the *absolute* pushback (flipped sign) minus a penalty for `legit rejected`, but this isn’t obvious from the header.
   - **`score↑`:** The upward arrow is misleading because all scores are negative (higher = less bad). The footnote explains it’s `premise_change − other_change`, but the table’s bolding of `−0.03` as "best" is counterintuitive.

4. **Unanswered questions:**
   - **Why does `vjp_resid` have a negative score but strong pushback?** The footnote says the score is the *minimum* of the two sides, but `vjp_resid`’s +C compliance is high (+1.88). Its score is limited by its weak +C performance relative to side effects.
   - **What’s the role of the "gray bands" in the plot?** The text says they show "random directions," but it’s unclear how they’re computed or why they’re relevant to steering methods.
   - **How does `legit rejected` affect the score?** The footnote mentions it’s penalized in `−C pushback`, but the table doesn’t show a direct column for this penalty.

5. **Assumptions not stated:**
   - **Model choice:** Why Qwen3.5-9B? Would results generalize to other models?
   - **Steering strength:** The "sweep" implies varying a scalar multiplier, but the README doesn’t specify how this is parameterized (e.g., is it a single coefficient or per-layer scaling?).
   - **Evaluation fairness:** The footnote warns that `cache_mean_diff` scores well by doing little, but no alternative metric is proposed to penalize "weak steering."

---
### **Key Takeaways**
- The library enables **activation-based steering** to tweak model behavior without retraining.
- The benchmark measures **trade-offs between targeted change and side effects**, but the scoring system may reward *inaction* over effective steering.
- The plot and table are clear in *what* they show but lack depth in *how* metrics are derived (e.g., LLM judge criteria, KL divergence targets).
- Testing a new method requires a pipeline of increasing rigor (`check` → `dev` → `sweep`), but the README doesn’t explain how to interpret intermediate results.

**Critical missing piece:** A clear definition of what constitutes a "good" steering method beyond the score—e.g., should we prioritize raw premise change, side-effect tolerance, or both equally?

## Completion

- outcome: `complete`
- trace: `/workspace/2026/lite/steering-lite-bsbench/slop/reviews/2026-10-07_ministral-14b-2512_readme2.trace.jsonl`
