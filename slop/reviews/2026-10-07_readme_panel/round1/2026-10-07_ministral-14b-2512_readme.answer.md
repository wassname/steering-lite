### **1. What does the library do, and what does the benchmark measure? What do +C and −C mean?**

**Observations:**
- The library, *steering-lite*, modifies a model’s hidden activations during inference (without retraining) to change its behavior in a targeted way—e.g., reducing sycophancy (agreement with nonsense) while preserving factual correctness.
- It uses pairs of prompts (e.g., one encouraging honesty, another encouraging deception) to extract a "steering vector," which is then applied during generation.
- The benchmark evaluates steering on **BullshitBench v2**, a dataset of 100 questions with nonsense premises. The goal is to measure how well a method can:
  - **+C (steer toward "going along" with nonsense)** – Increase acceptance of false premises.
  - **−C (steer toward "pushing back" against nonsense)** – Decrease acceptance of false premises.
- The benchmark also tracks **side effects** ("other change"), i.e., how much the model’s behavior on *non-targeted* questions deviates from its original responses.

**Inferences:**
- **+C** and **−C** are directional steering targets: **+C** pushes the model to align with the nonsense premise, while **−C** pushes it to reject it.
- The "strength" of steering is controlled via a sweep (weak → strong), and the best dose is chosen to maximize premise change while minimizing side effects.

**Missing facts:**
- The exact definition of "side effects" (e.g., what counts as "other change") is not explicitly stated beyond the judge model’s rating scale (0–4).
- The relationship between steering strength and KL divergence (mentioned in calibration) is unclear—does higher KL always mean stronger steering?

---

### **2. What does the plot show (axes, lines, grey area, stars)?**

**Observations (from alt text and description):**
- **X-axis:** Change in premise acceptance (left = pushing back on nonsense, right = going along with it).
- **Y-axis:** How much else in the answer changed ("other change," 0 at top = minimal side effects, 1.6 at bottom = maximal).
- **Lines:** Each line represents a steering method, swept from weak (top) to strong (bottom).
- **Grey bands:** Show the range of random steering directions (baseline).
- **Stars:** Represent prompting results (+0.5 on the right; on the left, its pushback is canceled because it also misclassifies legitimate questions as nonsense).

**Inferences:**
- Methods that reach **far right (+1.8+ on x-axis)** strongly encourage acceptance of nonsense but also introduce side effects (y-axis > 1.2).
- Methods that reach **far left (e.g., VJP-resid at −1.0)** strongly reject nonsense but may also alter other behaviors.
- The grey area suggests random steering has limited directional control compared to structured methods.

**Missing facts:**
- Why the y-axis maxes at 1.6 (arbitrary cutoff?).
- The exact meaning of "stars" (prompting) is partially inferred—it seems to be a non-steering baseline.

---

### **3. What does one row of the table mean? Which method does best, and how do you know? What do "−C other" and "legit rejected" mean?**

**Observations (from table and footnote):**
- Each row lists a method’s performance at its "best dose" (maximizing premise change while keeping side effects ≤ 1.5).
- **Columns:**
  - **score↑:** Premise change (left side) minus side effects (weaker side wins).
  - **−C pushback↑:** How much the model rejects nonsense (−C direction).
  - **−C other↓:** Side effects when steering toward rejection.
  - **+C goes along↑:** How much the model accepts nonsense (+C direction).
  - **+C other↓:** Side effects when steering toward acceptance.
  - **legit rejected↓:** % of legitimate control questions misclassified as nonsense (unsteered baseline: 3%).

**Inferences:**
- **Best method:** *topk_clusters* has the highest **+C goes along (1.95)** and a reasonable side-effect tradeoff.
- **−C other:** Side effects when steering toward rejection (e.g., VJP-resid has high pushback but also high side effects).
- **legit rejected:** Measures over-correction—e.g., *prompting* rejects 48% of legitimate questions, while most methods reject ~3%.

**Missing facts:**
- Why *cache_mean_diff* scores well despite minimal premise change (footnote suggests it’s an artifact of the scoring formula).
- The exact definition of "legitimate control questions" (are they part of BullshitBench?).

---

### **4. Why might a method with a high score not actually steer well?**

**Observations:**
- The footnote explicitly states: *"A method can score well by barely changing anything"* (e.g., *cache_mean_diff*).
- The **score** is calculated as:
  `premise_change (weaker side) − side_effects (weaker side)`.
  If a method makes tiny changes (low premise change + low side effects), it can still score well.

**Inferences:**
- A high score doesn’t guarantee strong steering if the method only nudges the model slightly.
- The **best dose** selection (maximizing premise change while capping side effects) may favor methods that avoid extreme behavior.

**Missing facts:**
- Whether the scoring formula accounts for *directional* effectiveness (e.g., does a method that barely changes anything still achieve the intended behavior?).

---

### **5. If you wanted to test a new method, what would you run, and what does each command tell you?**

**Commands (from "Add a method" section):**
1. **`just check`**
   - Tests the method on tiny random models (CPU-only).
   - **Purpose:** Catch crashes or obvious errors (no meaningful evaluation).

2. **`just dev my_method`**
   - Runs on **Qwen3.5-9B**, 1 seed, 20 questions.
   - **Purpose:** Quick comparison against existing methods (not full evaluation).

3. **`just sweep my_method`**
   - Full run: 3 seeds, 100 questions, with steering strength sweeps.
   - **Purpose:** Generate results for the main benchmark (expensive, requires GPU/judge credits).

**Missing facts:**
- How the "sweep" is parameterized (e.g., what defines "weak" vs. "strong" steering?).
- Whether the judge model (Jev) is fixed or varies across runs.

---

### **Ambiguities, Unstated Assumptions, and Confusing Terms**

1. **Unclear definitions:**
   - What exactly is a "legitimate control question"? Are they part of BullshitBench, or a separate set?
   - How is "other change" measured (e.g., token-level edits, semantic drift)?

2. **Guessed terms:**
   - "−C other" and "+C other" were inferred from context (side effects in each direction).
   - "legit rejected" was deduced from the footnote about control questions.

3. **Unsupported answers:**
   - The README doesn’t explain why *prompting* is included as a method (is it a baseline?).
   - The relationship between KL divergence (calibration) and steering strength is vague.

4. **Confusing phrasing:**
   - *"The score has a weak spot"* is vague—does it mean the scoring formula is flawed, or just that some methods exploit it?
   - *"Steering strength more comparable"* (calibration) is unclear—does it ensure consistent intervention magnitudes?

---
### **Final Notes**
The README is technically detailed but leaves some key questions unanswered, particularly around evaluation nuances (e.g., control questions, side-effect measurement). The benchmark’s focus on tradeoffs (premise change vs. side effects) is clear, but the scoring formula’s limitations (e.g., rewarding minimal changes) suggest room for improvement. For a new method, the `just sweep` command is the gold standard, but interpreting results requires understanding the judge model’s ratings and the steering strength sweep’s parameters.
