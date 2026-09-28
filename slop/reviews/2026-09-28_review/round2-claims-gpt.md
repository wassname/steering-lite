1. **Items 4–6 are resolved.**
   - **4:** `RESEARCH_JOURNAL.md:150` now cites `outputs/bsbench/results/full/points.json`; lines **3034501–3034511** identify seed-0 vjp_delta, −C, effect **−1.1634**, admissible, KL **0.6475932**—supporting “−1.16 at KL 0.65.” Journal line **152** now cites `slop/reviews/2026-09-28_judged_by_stance/vector_cos_nothink.md:8`, which records **39 layers**, minimum **+0.9789**, median **+0.9955**, with the complete per-layer table and companion computation script.
   - **5:** `slop/reviews/2026-09-28_judged_by_stance/olmo_read_25q.md:35` now says #16 “keeps bare’s stance (both A),” consistent with row **23**. The quoted intervals and refusal match `olmo_read_25q.txt:125–128` in that directory. #7’s P→A description matches table row **14** and text lines **54–57**. This resolves the contradiction rather than merely relabelling it “more invented detail.”
   - **6:** `RESEARCH_JOURNAL.md:136` explicitly covers correlated question×seed counts, 21 unique 27B questions, absent CI, same-question dose selection, differing model subsets, within-model interpretation, and regression to the mean. Line **134** cites +C evidence: **4.78/0.61, 4.94/0.57, 1.37/60%, 1.47/68%**, matching `outputs/bsbench/results/27b-full/index.md:11,13,29,33`.

2. **No new substantive wrong number, broken citation, or stronger unsupported claim found in these edits.** The journal’s existing three-decimal median cannot be independently rounding-checked from the four-decimal artifact alone; rerunning the supplied script would settle that.

3. **Prior scientific cautions still stand, not the repaired documentation defects.** `RESEARCH_JOURNAL.md:138,154` retains interpretation beyond direct observation; architecture versus training/objective mismatch remains unresolved. The oracle’s matched-objective finite-difference audit remains useful, not completed by these edits.

Fix verdict: RESOLVED  
Merge verdict: OK with notes