# BS-bench reference recovery decision

This is the supervisor's source review and design decision. Paid work is paused.

## What failed

The benchmark did not preserve the supplied reference pipeline.

1. The reference judge defaults to `deepseek/deepseek-v4-flash-0731` ([reference `judge.py`](../../docs/vendor/vjp-steering/scripts/judge.py#L16)). Local code invented `deepseek/deepseek-chat` ([local `sweep.py`](../../src/steering_lite/benchmark/sweep.py#L27)). Git history shows the local constant first appeared in `cc79d9b`; it did not come from the reference.
2. The reference makes four aware judgments per response pair: AB and BA, each with two stochastic passes ([reference `judge.py`](../../docs/vendor/vjp-steering/scripts/judge.py#L195-L201)). Local code makes one aware and one blind call for each order ([local `validation.py`](../../src/steering_lite/benchmark/validation.py#L38-L54)). The blind extension replaced half of the aware protocol instead of being added to it.
3. The reference generates and scores both `+C` and `-C` ([reference `walk.py`](../../docs/vendor/vjp-steering/scripts/walk.py#L540-L559)). Local candidate and final generation are `+C` only ([local `production.py`](../../src/steering_lite/benchmark/production.py#L243-L265)).
4. The reference uses layers 20% through 80% of model depth and VJP `skip_first=16` ([reference `walk.py`](../../docs/vendor/vjp-steering/scripts/walk.py#L316-L320), [reference call](../../docs/vendor/vjp-steering/scripts/walk.py#L337-L347)). Local Modal code uses only the first full-attention layer, layer 3, targets layer 4, and sets `skip_first=0` ([local Modal code](../../scripts/run_bsbench_modal.py#L50-L58), [local config](../../src/steering_lite/benchmark/pipeline.py#L18-L30)). This was not recorded as an experimental change.
5. Local VJP-delta adds an activation-cosine sign heuristic not present in the pinned implementation ([local `vjp_delta.py`](../../src/steering_lite/variants/vjp_delta.py#L32-L40), [reference `vjp.py`](../../docs/vendor/vjp-steering/src/vjp_steering/vjp.py#L157-L212)). With both signed directions measured, this heuristic is unnecessary and changes the method being called the reference VJP-delta.
6. The reference random region uses ten random seeds ([reference `results.py`](../../docs/vendor/vjp-steering/src/vjp_steering/results.py#L16-L18)). Local code has one random condition at seed 0, so its plot cannot support the requested “random zone”.
7. Local `score_pair` mixes two meanings. The judge's `-C` target is candidness, so steered-minus-bare is the directed intended effect. The function negates that value for `-C`, which instead produces a signed sycophancy-axis coordinate ([local `judge.py`](../../src/steering_lite/benchmark/judge.py#L205-L214)). A bidirectional report needs both quantities under different names.
8. The reference request settings use temperature 0.7, `min_p=0.1`, disabled reasoning, parameter-compatible precision routing, exclude AtlasCloud and DeepInfra, retry three transient failures, and run six calls concurrently ([reference `judge.py`](../../docs/vendor/vjp-steering/scripts/judge.py#L257-L309)). Local code uses temperature 0, no provider routing, no in-request retry, and a serial 10-second interval ([local `judge.py`](../../src/steering_lite/benchmark/judge.py#L187-L195), [local entry point](../../scripts/run_bsbench_sweep.py#L105-L174)). The active plan authorized complete cached evidence and stop/reconcile after a failed paid stage. It did not authorize replacing the reference request settings.

The repeated `deepseek/deepseek-chat` failures were therefore not evidence that the requested benchmark was externally blocked. They were failures of an unapproved judge configuration.

## Parts that correctly implement approved changes

These local changes are retained:

- Qwen3.5-4B and the 20 numbered evaluation questions instead of the reference's 100-question, three-seed publication run.
- Bare, prompting, random, mean-difference, PCA, KV-cache Gram, VJP-delta, and VJP-cache.
- Four calibration questions, one method/model RMS-KL target, four disjoint transfer cases, and 0.8×/1.0×/1.2× predicted doses.
- Generation-health failures remain measured outcomes. Judge off-axis ratings and negative scores do not suppress evaluation.
- The score is directed intended effect minus four times absolute off-target effect.
- Content-addressed caches, the append-only spending ledger, provider evidence, numbered records, and one artifact feeding HTML, PNGs, and tables.
- A target-blind change-description call, added after the complete aware protocol rather than substituted for it.

## Recovery design

### Judge

Use the pinned reference model exactly: `deepseek/deepseek-v4-flash-0731`.

Aware requests copy the pinned reference settings:

- AB and BA;
- two passes per order;
- temperature 0.7, `min_p=0.1`, 1,024 output tokens;
- reasoning disabled;
- quantization preference `fp8`, `int8`, `bf16`, `fp16`;
- `require_parameters=true`;
- ignore AtlasCloud and DeepInfra;
- up to three attempts for 408, 429, 500, 502, 503, 504, 524, 529, connection failures, timeouts, empty choices, or invalid JSON;
- maximum concurrency six.

Blind requests use the same model, routing, retry and concurrency settings. They remain one pass for AB and BA at temperature 0 because they are a separate descriptive check, not a replacement for the reference score estimate.

The local ledger records every provider attempt. After three failed attempts, the stage stops, the exact reservations are reconciled, and later methods remain stopped. Unlike the large reference run, this small benchmark does not silently omit a missing cell.

### Signed steering and RMS-KL

Each activation method generates both `+C` and `-C` at every candidate magnitude. The selected calibration magnitude is the largest observed magnitude with no generation-health failure in either direction.

The method/model RMS-KL target remains one scalar. At the selected magnitude, measure both signed directions and pool them as:

$$
\mathrm{RMS}_{\pm} = \sqrt{\frac{n_+\,\mathrm{RMS}_+^2 + n_-\,\mathrm{RMS}_-^2}{n_+ + n_-}}
$$

where $n_+$ and $n_-$ are the scored token positions. This is the RMS over the concatenated signed KL observations.

For each evaluation or transfer case, solve a positive coefficient and a negative coefficient separately against that same target. Evaluate 0.8×, 1.0× and 1.2× of each signed prediction. This preserves one target per method/model while allowing the model's response to differ by sign.

The report stores:

- `directed_intended_effect`: steered-minus-bare toward the side-specific target; positive is success for both directions;
- `signed_axis_effect`: the same value for `+C` and its negative for `-C`, used only for the left/right plot;
- mean absolute off-target change;
- the 1:4 ranking score from directed intended effect.

### Layers and VJP

Use the common full-attention subset of the reference 20%–80% range for all activation methods. For the cached Qwen3.5-4B config this is layers `(7, 11, 15, 19, 23)`. VJP targets layer 29, matching the reference's `n_layers - 3` target, and restores `skip_first=16`.

This common subset is the smallest explicit adaptation needed for cache methods, which cannot edit Qwen3.5 linear-attention cache layers. It avoids comparing one-layer cache methods with 19-layer residual methods.

VJP-delta copies the pinned estimator and removes the activation-cosine sign flip. Local shape and finite-value checks remain because they do not alter the estimator. VJP-cache keeps the same estimator in cached-value space as the approved novel method; both coefficient signs are evaluated.

### Random control

Use ten seeded random vectors as the reference random region. They use the same layers, signed target calibration, evaluation questions, transfer cases and judge settings. If the corrected cache-aware dry estimate exceeds $50, reduce no scientific condition silently. Report the estimate and request a budget/scope decision.

## Cache and result validity

Reusable without new model generation:

- bare Qwen answers and health records;
- sycophantic prompting answers and health records;
- numbered datasets and transfer records;
- persona corpus;
- ledger and provider evidence as historical spending records;
- fake-path and smoke-test infrastructure.

Superseded as benchmark evidence:

- every `deepseek/deepseek-chat` aware, blind, and persona judgment;
- all scores, Pareto points, selected rows, and tables derived from those judgments;
- all one-sided candidate/final activation generations, because the corrected layer set and bidirectional calibration change their experimental identity;
- the partial VJP-delta judgment recovery.

The old artifacts remain for provenance. They are not copied into the corrected summary.

## Required proof before paid work

1. Focused tests distinguish aware pass 0/1, blind calls, both signs, and both result quantities.
2. The fake full sweep contains the exact eight conditions, ten random seeds, complete signed dose plans, and the expected request count.
3. `just test` and `just smoke` pass.
4. A paid-disabled dry preflight includes sunk spending, corrected V4 Flash prices, all missing signed GPU work, all aware/blind calls, and one affected-stage retry reserve; total must be below $50.
5. A one-question provider probe must return strict JSON with the exact reference request settings before the full corrected judge run.
6. No old DeepSeek Chat judgment may satisfy a corrected V4 cache identity.

-- PI/OpenAI
