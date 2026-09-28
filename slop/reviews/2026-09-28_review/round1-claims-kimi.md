All checks complete. Review findings below.

## Review: 2026-09-28 journal entries vs cited evidence

**Verified correct (with repro):**

- **By-stance table**: all 36 n/gain numbers in the journal match `by_stance.md` exactly (regenerated logic: `by_stance.py` `g = -q["effect"]` = bare − steered premise, matching `results.py:80` `effect = steered − bare`; dose = `best[m]["-C"]` = Pareto-best from `results.py:214`; buckets ≥6/≤1; n = question×seed, e.g. 27B 63 accepts / 3 seeds = "21 of 100" ✓, 4B 153/3 = 51 ✓, OLMo 85/1 = 85 ✓).
- **vjp_split**: recomputed from `outputs/bsbench/*/vjp_split/vjp_delta_s0.json` — median `split_cos` 0.993/0.986/0.985 and median `cancel` 1.67/1.37/1.29, exact match.
- **OLMo walk claims** (`olmo-full/points.json`): vjp_delta -C admissible effects −0.006…−0.215, KL 1.93 at C=0.397, broken at C=0.5 → "−0.01 to −0.22 up to KL 1.9, then breaks" ✓; mean_diff −1.70 ✓; 4B vjp_delta seed 0 = −1.163 at KL 0.65 ✓.
- **nothink**: score −0.12 [−0.22, −0.00], on-axis/room +0.00 [−0.02, +0.05] match `results-olmo-nothink.log` exactly.
- **Correction table**: all 10 numbers exact vs `points.json` (78% ">" / 63 words at C=0.397 etc.); "only at its strongest dose" verified — 0% ">" at all 11 lower admissible doses; bare 33.7 ≈ 34 words ✓.
- **olmo_read_25q.md vs .txt**: spot-checked all 25 rows; readings consistent. Judgment calls I accept: #2 vjp_cache "A" (tautological restatement, could be P), #7 mean_diff "R" (gives "Zero" then refuses). Counts line (R 9/P 1; R 1/P 4; R 0/P 3) recomputed from the table — exact.
- **Mechanism claims**: vjp_delta averages gradients over all valid prompt positions (`vjp_delta.py:81-125`), mean_diff reads last token (`mean_diff.py:4`); `<think>` literal prefix, eval thinking off (`walk.py:70,100`) ✓. "ISO 32170"/"ABA Model Standard 4.7" examples exist in `answers/vjp_delta-nothink_s0/-C_C0.396*.jsonl` and do reword-and-accept ✓.

**Findings:**

- **P2** — "4B vjp_delta reaches −1.16 at KL 0.65" is cited parenthetically to `outputs/bsbench/results/olmo-full/points.json`, which contains no 4B data; actual source is `results/full/points.json`. Number itself is correct.
- **P2** — "Vector cos to the default vector per layer: min 0.979, median 0.995" appears in none of the three cited logs (`grep cos` hits only "cost"). I reproduced it from `vectors/vjp_delta{,-nothink}_s0.safetensors` (min 0.9789, median 0.9955), so the claim is true but the citation doesn't contain it.
- **P2** — `olmo_read_25q.md` note "VJP is worse than bare … #16 for vjp_cache" contradicts its own table row (#16: bare A, vjp_cache A); "worse" refers to fabricated precision, not the A/P/R stance scheme the file defines.
- **P2** — By-stance interpretation: conditioning on bare stance is valid (pre-treatment variable), and the +C side actually supports "VJP works where there is room" (27b-full/index.md:13,33: vjp_delta +C on +4.78, blind stance +1.37, P(accepts) 60%) — though the entry never cites it. Residual weakness: the accepts bucket is 21 unique questions (63 correlated q×seed pairs), no CI, and vjp_delta's +1.41 vs mean_diff's +4.28 on identical questions makes "probably still works (0.75)" generous for vjp_delta specifically; the hedging ("probable", "mean_diff's lead is real") makes this acceptable.

No P0/P1 issues: every checkable number matches its source, the correction entry is fully reproducible, and interpretations are labelled with calibrated confidence.

Merge verdict: OK with notes