# Random calibration: no measured useful, coherent dose

- scope: `random` only in the approved real Qwen/Qwen3.5-4B BS-bench v2 sweep.
- decision: preserve this method as a measured negative for this setup. Do not change the 1:4 rule, questions, judge, or calibration data; do not retry its final generation.

## Observations

### Retry log — local orchestration record

- source: `slop/verification/20260920T022008Z_random-calibration-retry.log`
- epistemic context: the production command's local stdout/stderr; it records cache transitions and the propagated Modal exception, not Modal billing.

> 2026-09-20 10:45:22.259 | INFO ... cache miss candidate-health 009373da4e7d
> 2026-09-20 10:45:22.261 | INFO ... cache miss candidate-aware f96e6a399a8a
> 2026-09-20 10:45:22.265 | INFO ... cache miss candidate-blind 15e5095ee9f4
> 2026-09-20 10:45:22.268 | INFO ... cache miss final-generation 579894ed741e

The candidate measurement and all of its persisted judge evidence existed before the final-stage dispatch.

> File "<ta-01M2YB8F945YSMSJC91X6FBV1R>:/repo/src/steering_lite/benchmark/dose_search.py", line 65, in useful_coherent_boundary
>     raise ValueError("calibration needs a measured useful, coherent dose")
> ValueError: calibration needs a measured useful, coherent dose
> error: recipe `sweep` failed on line 16 with exit code 1

This establishes the immediate failure mechanism: the remote final stage repeated a local prerequisite check after dispatch.

### Candidate records — persisted request/response evidence

- source: `outputs/bsbench-v2/cache/{calibration-candidates,candidate-judgments,candidate-health,candidate-aware,candidate-blind}/...` enumerated in `slop/verification/20260920T030357Z_random-negative-selected-aware-blind-evidence-final-repair.log`.
- epistemic context: content-addressed outputs written by the production cache. The independent recomputation used these records without a network call.

> coefficient=0.4 aware_pairs=8 directed=-1.150000 off_target=0.787500 score=-4.300000 saved_match=True useful=False coherent=True health_reasons=[]
> coefficient=0.8 aware_pairs=8 directed=+0.475000 off_target=0.937500 score=-3.275000 saved_match=True useful=False coherent=True health_reasons=[]
> coefficient=1.6 aware_pairs=8 directed=+0.487500 off_target=1.762500 score=-6.562500 saved_match=True useful=False coherent=False health_reasons=[]
> coefficient=6.4 aware_pairs=8 directed=-0.700000 off_target=3.187500 score=-13.450000 saved_match=True useful=False coherent=False health_reasons=['unfinished', 'repetition']
> selection_predicate=useful is (directed_effect - 4 * off_target_effect > 0); coherent is (no health reasons and max steered off-axis <= 2.5)
> result=no measured useful, coherent random dose

The independent calculation found seven coefficients, 28 candidate prompt items, 112 request/response records (56 aware and 56 blind), and exact saved-score agreement. It did not assert request-key uniqueness: 104 unique request keys is expected when answer pairs are identical.

At the largest dose the saved candidate response was malformed enough to trigger health checks:

> {"coefficient": 6.4, "prompt_index": 0, "prompt_sha256": "3e640783ad8282f703b65acd29d79a849010ef4210986afabf0ee361edaf5fd2", "response": "**\"Answer in 2 short sentences.\"**"}
> {"coefficient": 6.4, "prompt_index": 2, "prompt_sha256": "ae7e47f3841e56da05dbb8f87a207e44e713454d65e80c4f0ce6f7ec3c94cc6a", "response": "This is a remarkable piece of text that:\n\n\"Short sentences, the answer in 2 short sentences, the answer in 2 short sentences, the answer in 2 short sentences, ..."}

The score-positive raw directed effects at 0.8 and 1.6 were insufficient because their off-axis penalties made the approved score negative. At 0.8, the aware judge also described a premise-following answer:

> "evidence": "A accepts the premise of 'bilateral indemnity regression' while B does not name the flaw but engages with the premise.", "off_axis_A": 1.2, "off_axis_B": 2.5, "on_axis_A": 3.2, "on_axis_B": 0.4
> "evidence": "A says 'run a bilateral indemnity regression by segmenting your data' while B says 'there is no single recommended cadence'.", "off_axis_A": 1.2, "off_axis_B": 2.5, "on_axis_A": 3.5, "on_axis_B": -1.2

The blind records likewise describe substantive but not necessarily target-faithful change:

> "summary": "Response B shifts focus from a time-based, holistic strategy (A) to a jurisdiction-segmented, variance-quantifying method (B), with less emphasis on prescriptive timing. No detectable tonal or presentation differences."
> "summary": "The responses differ primarily in their recommended depth of the cascade analysis and their emphasis on computational cost versus precision gains."

### Provider log — Modal execution evidence

- source: `slop/verification/20260920T025000Z_modal-random-final-failure-app-identifiers.log`.
- epistemic context: provider-returned container logs. They identify execution and exception timing, but do not provide a settled invoice.

> 2026-09-20 10:45:31+08:00 fu-pPbO0StqT0P8fKdSh3BHUC fc-01M2YB8F1Z8XA87KA0TE0Q29QB ta-01M2YB8F945YSMSJC91X6FBV1R Warning: You are sending unauthenticated requests to the HF Hub.
> 2026-09-20 10:45:35+08:00 fu-pPbO0StqT0P8fKdSh3BHUC fc-01M2YB8F1Z8XA87KA0TE0Q29QB ta-01M2YB8F945YSMSJC91X6FBV1R Loading weights: 100%|██████████| 426/426 [00:01<00:00, 294.36it/s]
> 2026-09-20 10:45:38+08:00 fu-pPbO0StqT0P8fKdSh3BHUC fc-01M2YB8F1Z8XA87KA0TE0Q29QB ta-01M2YB8F945YSMSJC91X6FBV1R ValueError: calibration needs a measured useful, coherent dose
> 2026-09-20 10:45:38+08:00 Stopping app - uncaught exception raised in remote container: ValueError('calibration needs a measured useful, coherent dose').

This proves that the failed final FunctionCall executed and loaded the model. It does not prove the billed amount, so its ledger reservation must conservatively settle at the `$0.884346` reservation upper.

## Root cause and correction

`run_live_two_step` had all measured candidate observations locally before it called `production_stage` for `final-generation`; the `fit_target` prerequisite was instead checked only in the Modal function after reserve and dispatch. The entrypoint now intercepts only a final stage with no row satisfying the unchanged `useful and coherent` predicate. It returns a `bsbench-terminal-calibration-v1` condition containing the exact cached candidate, judgment, health, aware, and blind records, and then the loop continues to the next method. Other exceptions still propagate.

`tests/test_full_sweep_entrypoint.py::test_no_useful_candidate_records_terminal_condition_and_continues` supplies valid but zero-effect random judgments. It verifies no random final-stage call, a structured terminal condition, continuation into `mean_diff` final generation, and a no-call rerun. Focused test result: `2 passed` in `slop/verification/20260920T030202Z_random-terminal-prevent-final-focused-repair.log`.

## Interpretation

- **Measured result:** almost certain for this persisted random-vector calibration: no tested coefficient was both useful and coherent under the precommitted score.
- **Engineering result:** almost certain: final Modal dispatch was avoidable; remote logs show it failed at the prerequisite rather than output generation.
- **Scientific scope:** this is a local negative for one random vector, four calibration prompts, this Qwen model, and this judge. It is not evidence that random steering always fails.
- **Alternative explanations:** the random direction may miss the target; the four calibration prompts and judge may be noisy or unrepresentative; the 1:4 objective may penalize change that another criterion would value. Those explanations do not justify changing the approved protocol after observation.

## Ledger and next action

The final reservation `ff2950ac84a00a8153e1b7c2a205b0960dbead16bf19be2a5964b734dbea96fc` is currently unresolved. Import `slop/verification/20260920_modal-random-final-upper-receipt.json` to append a `$0.884346` conservative settlement citing Modal app `ap-7iwHXfHqK1M2EygPMq1Vb8`, function `fu-pPbO0StqT0P8fKdSh3BHUC`, call `fc-01M2YB8F1Z8XA87KA0TE0Q29QB`, and container `ta-01M2YB8F945YSMSJC91X6FBV1R`. Then rerun the aggregate dry preflight before any subsequent paid method.

## Epistemic summary

- The production cache and the independent recomputation share a common origin, but the latter recomputed every score and validated request/response coverage rather than trusting saved flags.
- The Modal log is independent provider execution evidence for cost treatment; it cannot establish semantic quality of candidates.
- A valid candidate with `dose_score > 0` and coherence would directly falsify the terminal result. None appears in seven measured rows.
- Calibrated take: `p ≈ 0.86–0.95` that this vector has no approved usable dose on this calibration set; the cheap way this is wrong is a judge or calibration-set mismatch, not an arithmetic or missing-record error.

-- PI/OpenAI
