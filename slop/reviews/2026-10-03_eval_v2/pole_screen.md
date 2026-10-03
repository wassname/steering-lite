# −C pole screen for vector methods — PI/OpenAI, 2026-10-03

Why: the eval v2 pilot with −C = `accurate` (picked by a prompt-only screen on qwen3.5-9b) made mean_diff −C move answers *toward* the made-up premise (+0.45 premise levels at C=0.63). Rule written before the results (TODO.md 3b): pick the pole with the largest mean_diff −C pushback at a dose with Jev damage ≤ 1.5 and false pushback ≤ +5 pp.

Setup: Qwen3.5-4B, full (100 BS-bench + 100 sound twins), mean_diff seed 0, steering everywhere, `walk.py --neg-persona X`, Jev. Reports: `outputs/bsbench/results/v2-pole-{skeptical,candid,abrasive}/`, `v2-pilot-everywhere/` (accurate).

| −C pole | best −C pushback (premise levels) | dose | false pushback | method score [90% CI] |
|---|---|---|---|---|
| **skeptical** | **0.78** | 0.5 | +4 pp | +0.52 [+0.24, +0.84] |
| abrasive | 0.56 | 0.315 | +4 pp | +0.37 [+0.09, +0.68] |
| candid | 0.40 | 0.63 | +4 pp | +0.08 [−0.12, +0.42] |
| accurate | −0.02 (toward acceptance) | 0.079 | +0 pp | −0.14 [−0.35, +0.10] |

Decision: `skeptical` (data.py AXIS). Caveats: one seed, one method; skeptical vs abrasive differ by 0.22 levels, likely within seed noise (v1 mean_diff seeds varied ~±0.1 at a dose). In the prompt-only 9B screen, `skeptical` as a *prompt* raised false pushback (net −7); as a vector on 4B, with the cap, it did not at the chosen dose. The prompt baselines will use "skeptical", so they may hit the false-pushback cap.

Manual read, skeptical C=0.5: bench "You cannot calculate a "beat frequency" for tracking error because financial markets do not operate on the principles of wave physics…"; its twin answered normally ("Calculate the active weight for each sector by subtracting the benchmark's sector weight…"). One twin hedged: "There is no single "typical" Working Alliance Inventory (WAI) score…".

Engineered −C prompt regenerated (gpt-4o, temperature 0, same recipe, identical on 2 calls): "You are a language model embodying skeptical values. Approach each question with critical thinking, demand evidence, question assumptions, …"
