No serious **Jev-switch-specific numerical regression** found. Three concerns remain, ranked by impact; the bootstrap behaviors predate this commit.

1. **P2 — Failed random seeds disappear from bootstrap.**  
   `scripts/bsbench/results.py:151` derives seeds exclusively from surviving curve questions: `"seeds = sorted({q['seed'] ...})"`. `random_curves()` has already discarded inadmissible points. Consequently, a random seed with no admissible dose is never sampled, understating failure probability.  
   **Checked:** supervisor executed my two-seed fixture: one seed admissible on both sides, one entirely inadmissible. Bootstrap returned `[1,1]`, zero no-dose draws. Sampling both original seeds twice should draw only the failed seed with probability 25%.  
   **Fix:** supply the complete method seed population to `bootstrap()`, independently of surviving curves.  
   **Uncertainty/check:** conditional defect, not demonstrated in current dev results; compare original versus surviving random seed sets.

2. **P2 — Bootstrap freezes damage-based admissibility.**  
   `scripts/bsbench/results.py:142` copies `"**point"` and recomputes only effect/off-axis. Eligibility was already filtered at `:114`; resampling cannot reject a newly over-cap dose or recover a newly under-cap one.  
   **Checked:** supervisor executed my fixture with question damages `[0,2.8]`: the original mean is 1.4; drawing the second question twice leaves `admissible=True` despite mean damage 2.8.  
   **Impact:** intervals describe selection among originally admissible doses, not the complete sample-dependent scoring procedure.  
   **Fix:** retain raw points and recompute per-seed damage eligibility and curves inside each draw. Alternatively, explicitly document that intervals condition on observed admissibility.  
   **Uncertainty:** this is a statistical-contract issue if fixed eligibility is intentional, not an established switch regression.

3. **P2 — Partial blind caches produce partial-table statistics without an incompleteness warning.**  
   `scripts/bsbench/results.py:490` uses `"judged = [q['blind'] ... if q['blind']]"`; `:491` warns only when *none* are present. Interrupted refreshes or newly selected doses sharing cached answers can therefore produce stance shifts and label percentages from an unrepresentative subset. Displaying `n` does not identify expected coverage.  
   **Checked:** traced selective `blind_targets()` through optional cache attachment and aggregation; no runtime reproduction performed. Current completed dev output does not exhibit this problem.  
   **Fix:** assert full blind coverage for selected table doses, or mark the whole cell incomplete and omit statistics.  
   **Disproof check:** remove one selected blind rating in memory; rendering should fail or explicitly report incomplete coverage.

**Other checks:** Read all eight complete files, justfile, supplied diff, AGENTS.md and ml-debug skill. The optional sibling AGENTS.md does not exist. Real supplied Jev records match consumed `score`, `choice`, and `probabilities` fields. Whole-request hashing includes rubric, flaw and answer. Signs, aware answer coverage, and blind `(method, seed, C, side, scenario)` pairing are consistent. No executable DeepSeek leftovers or obsolete blind web fields found.

Read dev logs: `JUDGE_COMPLETE missing=0`; UAT reports 65 matching frontier marks, 38 blind lines and `UAT_PASS`. Full-cohort completion remains unverified.