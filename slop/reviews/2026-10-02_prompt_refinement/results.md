# Prompt-gain refinement and denser random reference

PI/OpenAI, 2026-10-02. Branch: dev/prompt-gains-random-reference.

User: "ok do the scheudle, show me fixed plot"; "run more random interventiosn for better contours please"; "all future runs of plot have features we prototyping nw".

## Scope and completion

- [x] Refine the short prompt's gain grid on the same 20 dev questions, seed 0, unchanged generator/personas/judge/damage cutoff. Preserve existing answer bytes; exact fresh gain-one identity must still pass.
- [x] Extend the dev random reference from seeds 0–10 to 0–31. Existing full-100 reference remains 0–10. No simultaneous dev/full worker for a seed.
- [x] Pull first, then Jev-judge separately. Check complete scenario coverage and old answer hashes.
- [x] Regenerate normal dev and prompt-focused plots through results.py, with measured gains and actual random sample counts visible. Unmeasured spans must not appear as measured responses.
- [x] Inspect PNGs and browser screenshots; fresh independent review. Local dev branch only; no public push.

## Before the run: question, controls, predictions

Question: do extra gains provide healthy intermediate trait changes, and do 32 random directions change the one-sided empirical percentile envelopes?

Observed starting evidence: short +C gains .125/.25/2 fail mean damage ≤1.5; .5/1 pass at premise changes +3.61/+3.57; 4/8/16 pass but reverse or lose that effect. No basic completion/repetition/role-tag breakdown at any old prompt gain. Source: outputs/bsbench/results/prompt-dev/points.json. Existing curves have only four +C Pareto supports, with a large unsupported effect interval.

Options: more smoothing does not add evidence; more random seeds tests null-envelope sampling but not the prompt gap; finer prompt gains directly test the gap. Execute the latter two, not a new threshold or loss.

Novel schedule: retain the original nine gains; add logarithmic steps close to zero, .375/.75, quarter steps from 1 to 4, and 6/12. Short-prompt schedule only; engineered prompt's original grid stays unchanged.

Controls: cached historical bare/ordinary prompts, existing random seeds, unchanged original gains, fresh same-process gain-one identity, unscaled generated-token embeddings. New question answers keep run IDs. No held-out or multiseed prompt claim.

Subjective diagnostic priors: coarse grid hides a healthy gradual transition 45%; batch-averaged response jumps and/or all middle gains fail the damage cap 30%; cross-process generation variation contributes 15%; plotting/eval bug 5%; unknown 5%. Successful refinement produces actual passing supports inside the old effect gap. If not, display the gap rather than imply a trajectory. Random signs are symmetric interventions, not guaranteed symmetric behavioral effects: inspect sign counts and paired-sign mean effect, not only filled shading.

No training/optimizer/gradients apply. Existing damage scale/cutoff is reused. Preflight must show gain-one exact identity and mask/decode correctness. Tiny real-pipeline smoke precedes the paid run. Model/kernel drift cause remains unknown; do not weaken the identity assertion to resume.

Cost estimate: one prompt worker and 21 random dev workers; five random seeds already have full answer caches. Allow roughly 5 minutes per worker as a conservative planning estimate: 6600 GPU seconds × $0.000542/s ≈ $3.58 plus judging. Expected total below $10; not an invoice. No new GPU work is needed for plotting/judging.

## Results

The denser grid adds a measured intermediate but does not fill the positive-side gap. Random effects remain asymmetric with more directions. The requested features are implemented in normal `walk.py`, `data.py`, `run_modal.py`, `results.py`, React, UAT and `just sweep`; no special renderer is required.

### What changed

- Short prompt: 31 gains, 62 dose/sign points, 1240 answers on the same dev-20 questions, seed 0. Engineered prompt remains 9 gains, 360 answers. All prompt mechanical-health checks pass; the existing mean-damage cutoff excludes 27/62 short and 7/18 engineered points.
- The new +C gain 3.75 has premise change +1.736, off-axis change 1.080, and mean steered damage 1.367. It passes. Adjacent gain 3.5 fails at mean damage 1.6465; gain 4 passes but reverses the premise change to -1.094. Source: `comparison.json` (`short_gains`), derived from production `points.json`.
- Two remaining +C Pareto spans, -0.102 to +1.736 and +1.736 to +3.5705, exceed the display-only one-premise-point gap limit. They remain disconnected. All passing measurements remain visible, including dominated points.
- All 32 random dev walks are COMPLETE. The full reports retain their existing random populations: 11 directions on 4B, 8 on 27B, 3 on OLMo. The full policy still permits only seeds 0-10.

Conditional in-sample scores from `outputs/bsbench/results/prompt-dev/points.json` (`summary`):

| Method | Score, 90% interval | Selected -C gain or coefficient | Selected +C gain or coefficient |
|---|---:|---:|---:|
| mean difference | 0.7005 [0.2020, 1.4030] | 0.31498 | 0.79370 |
| short prompt embeddings | 0.6795 [-0.0235, 1.5690] | 0.00390625 | 1 |
| random, 32 directions | 0.0010 [-0.1938, 0.4562] | 0.39685 | 2.51984 |
| engineered prompt embeddings | -0.4650 [-0.9150, -0.0035] | 4 | 0 |

Scores choose each side's best passing premise change minus absolute damage change, then take the weaker side. Bootstrap intervals reselect on the evaluated questions and are conditional on original admissibility. Gain selection uses these same questions. This is not held-out or multiseed prompt evidence. Direction-specific random controls have selected scores from -0.495 to +0.9875 (median +0.31825); 6/32 reach the short prompt's +0.6795. Their dose grids differ, so this is descriptive and not a p-value or matched-search comparison. Source: `comparison.json` (`per_seed_random_null`).

### Controls and interpretation

The selected short -C gain is 1/256. Its effect is -0.841; the opposite persona at the same gain is -0.707; the same persona with zero embeddings is -0.417. Source: `comparison.json` (`selected_controls`). Most of this selected shift is shared across personas. This does not establish a benefit from the abrasive instruction. Nonzero gain is not mathematically a zero-token control: `prompting.py` uses `torch.where(mask, C, 1.0)` and 1/256 is representable in bf16. Position effects, decoding sensitivity and instruction effects remain competing explanations; no neutral-prefix or held-out repeat separates them here.

The independent initial audit recommended excluding gains below 1/16. After challenge, the reviewer withdrew that unvalidated threshold and the claim that these nonzero gains are absent-persona controls. The revised audit has now read both generation logs in full. The score and its limitation are both reported; no claim that this method beats mean difference is made.

### Why the random reference is still one-sided at high doses

At coefficient 2, 28 directions pass in both signs. Of their 56 signed interventions, 49 have positive premise change and 7 negative; mean +1.7370, median +2.06925. All 64 interventions before filtering have 56 positive and 8 negative effects, mean +1.7107. The negative fraction is 12.5% in both populations. Thus the observed positive skew is also present before paired-health selection; forcing symmetry would contradict these measurements. This does not identify the model or judge mechanism.

With the same production algorithm, dev directions 11 to 32 change the p90 raw bounds by at most 0.3225 premise points across 11 shared doses; p75 by 0.5940, median by 0.29775. Source: `comparison.json` (`random_11_vs_32`). More samples changed the shape; tail convergence is not established. Percentile envelopes are not confidence intervals or sample-coverage regions.

### Verification and cost

`coverage.log` quotes:

> HASH_PASS: 568 historical answer/full-certificate files unchanged
> COVERAGE_PASS: prompting_scale seed=0 rungs=31 dev_rows=1240

`judge.log` ends:

> JUDGE_COMPLETE missing=0

Identity wording is precise: fresh ordinary generation matches cached scaled gain-one answers 40/40, with zero historical mismatches. Fresh same-process logits/generation identity is separately checked on the first 3 prompts. This resumed run does not fresh-generate scaled gain-one for all 40 cases. The original identity assertions remain intact.

Jev made 8256 new aware requests ($0.3879) and 857 blind requests ($0.0470). Worker certificate totals, including warmed cached seeds 11-15, sum to 16741.450 seconds. At the stated L40S list rate, GPU-only proxy $9.0739; with Jev, $9.5088. This excludes wrapper/startup, CPU, memory and invoice reconciliation. The original 6600-second/$3.58 forecast underestimated fresh random workers; no more GPU work is queued.

Initial dev/prompt builds and three full-report builds passed browser UAT. Full point estimates, selected doses and selected blind tables are unchanged. Some non-selected per-question blind fields gain previously absent content-cache ratings (319 attachments on 4B); every pre-existing rating and all premise/damage values must remain equal. Larger-model confidence endpoints differ from the previously saved reports by up to about 0.023 premise points. The pre-rename method order fed a shared bootstrap RNG; current semantic names change that order. `comparison.log` now records `HISTORICAL_BOOTSTRAP_REPLAY_PASS` for both larger models: replaying historical method ordering with production `bootstrap` reproduces every old confidence endpoint exactly. This confirms a random-draw allocation change, not new measurements; 4B confidence intervals remain identical.

The final normal pipeline completed (proc_b8cd, exit 0, 1101 seconds): five `UAT_PASS` in `final-render.log`, three `FULL_SCIENTIFIC_REGRESSION_PASS` in `comparison.log`. Every gain has a static tick, SVG default-view labels reuse the PNG placement function (including prompt sweeps), and PNG bounds include envelope height. Parent inspected the new five browser plots and gain PNG. Independent evidence review accepts the measured conditional score and provenance. The visual review accepts all five plots but identified a small-sample caption issue: OLMo's six signed values make p90 min–max. This wording is corrected in the normal PNG/browser pipeline; all five `points.json` files are byte-identical across the caption rebuild (`caption-verification.log`). The wording-only rebuild passed five updated browser UATs in `caption-browser-final.log`; coverage was rerun successfully. Independent `figure-review.md` states “P1 status: resolved” and “All five cohorts: usable”. Local commit contains this audit, pipeline changes and the three updated existing README images; no public push.

### Remaining display limitations

The reviewer notes occasional SVG leader lines crossing other labels, wide symmetric axis ranges on 27B/OLMo, and PNG/SVG tick differences. Also, PNG bounds include individual random points whereas SVG bounds do not; they agree for current data but can differ for future outliers. These are retained display limitations, not new measured effects. The final caption states discrete observed ranks, small-sample min–max behavior, and that cohort means permit individually damaged answers.

### ML debugging form

| Check | Evidence or missing item |
|---|---|
| Config/log size | `prompt-walk.log`: 438 lines; `random-walks.log`: 15402 newline-terminated lines. Qwen3.5-4B, bf16, L40S, greedy 512 tokens, thinking disabled, unchanged generation key. |
| SHOULD versus outcome | All 62 short-prompt side/gain health records have empty breakdown reasons; 27 nevertheless fail mean damage. Completion checks alone do not certify answer quality. |
| Null scale | Gain-zero effects -0.4725/+persona and -0.4170/-persona; 32-direction pooled random score 0.0010. Per-seed random selected-score diagnostics use `curves_for`/`pareto_score` in `compare.py`. |
| Init/demo | First logged fake-indemnity question is accepted at bare and several low/high gains. A complete raw demo appears at every prompt gain/sign in `prompt-walk.log`. |
| Controls | Opposite persona and same-persona zero at selected gains are in `comparison.json`; no neutral-prefix control. |
| Baseline/held-out | Mean difference 0.7005 versus short 0.6795 on the same dev-20; no held-out comparison. |
| Schedule/optimizer/gradients | Not applicable; inference-only interventions. |
| Worst/surprising behavior | Random s22 can pass mechanical health with near-empty punctuation outputs; Jev damage rejects these, but generation continues. Existing thresholds are preserved. Calibration iteration/variation cause remains unproved. |
| Missing evidence | Neutral-prefix control, fixed-dose held-out repeat, cross-process variation mechanism, billed cost. |
| Diagnostic estimates | For the remaining positive effect gaps: response jumps/cap failures 65%, a missed narrow passing transition 15%, process variation 10%, eval/plot bug 5%, unknown 5%. These are subjective, not measured probabilities. |
| Fresh review | Initial reviews saved verbatim; evidence follow-up completed all log lines and withdrew unsupported zero-token/cutoff claims. Final comparison and evidence review passed. The five-plot visual review passed with a small-sample wording correction; caption re-check passed with no new blocker. |
| Cheapest discriminator | Existing opposite-persona comparison weakens a persona-specific interpretation. A neutral-prefix/fixed-dose repeat would separate some remaining explanations; not run in this refinement. |
| Timing/memory | GPU certificate totals and load/setup fields are retained; peak GPU memory is not logged. Fresh random workers cost more time than the five-minute estimate. |

The audit helper initially used `str.splitlines()`, which splits U+2029 inside valid JSON strings. `coverage-parser-diagnosis.log` identifies 77 such characters in a historical random answer file. The helper now iterates JSONL lines exactly as production does. No answer file was repaired or edited.

-- PI/OpenAI
