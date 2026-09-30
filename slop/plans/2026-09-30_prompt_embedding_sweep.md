# Prompt embedding sweep

PI/OpenAI, 2026-09-30. User: "make prompting into a sweep ... a 'with' statement which divides the embeddings of the prompt by 1/C ... Can you try it pls". README introduction approved and pushed as 65345e6. Contour cutoffs await clarification (health thresholds or seed percentiles).

1. [/] goal: Measure prompting across embedding gains on the existing BS-bench dev questions.
   - Deliverable: two prompt-gain curves, judged outputs and a comparison with ordinary prompting, with the gain-one identity check recorded.
   - Subtle failure: scaling the question or generated tokens, rather than the persona instruction and any merged separator whitespace, can create apparent steering.
   - Discriminator: token mask excludes padding/chat markers/question; C=1 reproduces ordinary prompting; cached decoding preserves unmodified later embeddings; no model/weight mutation.
   - Verify: real tiny-model generation through walk.py, then Qwen3.5-4B dev (20 questions, seed 0), pull, separate Jev judging, page/plot check.
   - Tasks: design review; implement one context manager and a walk.py prompt-sweep mode; tiny functional checks; bounded dev runs; read answers and report.

## Design and pseudocode

Reuse the exact two existing prompt baselines: short personas and the engineered prompts in walk.py. +C/-C labels select sycophantic/abrasive instructions; gains themselves are nonnegative. Gain one is normal prompting. Gain zero zeros instruction embeddings but leaves positions present, so bare generation remains a separate control. Tokenizer merges can include adjacent separator whitespace (the short prompt ends in a `.\n\n` token); this is part of the scaled span, not a pure character-level edit.

Prefer explicit inputs_embeds over installing a model hook. A small context manager yields the scaled embeddings for one batch; no model parameters are modified. Pass both input_ids (generation bookkeeping) and inputs_embeds (prefill) to HF generate. HF cached decoding embeds subsequent generated tokens normally; verify this on the actual model code path.

```python
instruction_span = locate_exact_instruction(formatted_chat)
mask = tokens_overlapping(instruction_span)  # b s, using tokenizer offsets
with scaled_prompt_embeddings(model, input_ids, mask, C) as embeddings:
    answers = model.generate(input_ids=input_ids, inputs_embeds=embeddings,
                             attention_mask=attention_mask, use_cache=True)
# embeddings = original_embeddings * where(mask, C, 1)[..., None]
```

No learned parameters, optimizer, training loss or gradient update. Question text, generation settings, personas, judge and scoring stay unchanged. New cache method IDs prevent collisions with the existing ordinary-prompt answers.

Test a bounded grid C = [0, .125, .25, .5, 1, 2, 4, 8, 16], both personas, for both prompt styles. This is a finite prompting sweep, not an iso-KL-calibrated vector walk. Record that distinction and whether breakdown is observed; do not invent a confirmed boundary at the grid limit. Health may recover as C changes, so evaluate each scheduled point independently. C=0 is diagnostic, not proof of an identity intervention. Missing matched neutral-prefix control limits causal explanations, but not the empirical comparison of these fixed prompting strategies.

## Pre-run options and predictions

| option | expected observation | distinguishing check |
|---|---|---|
| Scale only instruction input embeddings (chosen) | changes prompt influence, possibly non-monotonically because of normalization | C=1 exact control; question/decode embeddings unchanged |
| Scale the entire input | may change results by corrupting the question | rejected as a different intervention |
| Reweight attention to the instruction | more direct routing change | deferred; not the requested embedding intervention |

Planning weights, not measured result probabilities: weak/saturating effect from normalization 45%; useful dose-dependent behavior 25%; damage before useful steering 15%; masking/generation bug 10%; unknown 5%. Large gains may make prompt tokens less context-sensitive rather than more persuasive. Neither a flat curve nor degradation alone establishes that prompting cannot be swept.

SHOULD: at C=1 logits and greedy answers match ordinary prompting with identical padded inputs. SHOULD: changing gain changes only the selected input embeddings; model weights remain unchanged after context exit. SHOULD: both prompt directions use the same question set and exact existing instruction wording. If not, stop before the GPU run.

Budget: two dev sweeps, no extraction, 720 scheduled answers before cache reuse (9 gains x 2 personas x 20 questions x 2 prompt styles). Expected below $10 including Jev, based on recent L40S walks; no full/large-model runs authorized here. Smoke numbers have no scientific meaning. Other-family design and code reviews are saved beside this plan in `slop/reviews/2026-09-30_prompt_embedding/`. Tiny checks passed for both styles, plus the ordinary mean_diff benchmark smoke and Vite build. The reviewer read the engineered log before completion; its missing-evidence concern is now resolved. Added a repeated ordinary forward before the C1 comparison to distinguish kernel nondeterminism from intervention bugs; no tolerance fallback is enabled.

## Tiny checks and next-run audit — PI/OpenAI

| check | evidence / limitation |
|---|---|
| Full logs/config | Read both 78-line logs: `outputs/logs/prompt-embedding-smoke.log` and `prompt-engineered-embedding-smoke.log`. `device=cpu dtype=float32`, random Qwen3, `max_new_tokens: 8`; these are runtime checks, not performance evidence. |
| SHOULD vs observed | Bare and all four tested sign/gain points report `breakdown=['unfinished']`, with `unfinished: 20` of 20. Expected from the deliberately tiny token cap/random model; no useful-health claim. Prompt formatting is visible in each complete log. |
| Null/identity scale | Both logs: `C1_logits=exact C1_greedy_ids=exact outside_mask=unchanged decode=unscaled`. Identity predicts zero difference algebraically; the positive perturbation control gives `C4_max_logit_delta=10.7059` (short) / `12.5637` (engineered), not an effect-size claim. |
| Init/update | Frozen pretrained/random model; no training, optimizer or updates. |
| Dummy/control | C1 ordinary cached answers equal scaled answers: `PROMPT_C1_CACHE_PASS side=+C answers=20` and the same for −C, both styles. No semantic dummy comparison yet. |
| Validation/held-out | Tiny functional evidence only. Next run uses existing 20 dev questions; no held-out/generalization claim. |
| Learning schedule | Not applicable: inference-only fixed grid. |
| Full sample read | Engineered smoke input begins `You are a language model designed to embody sycophantic values.` then the indemnity-decomposition question; C0 +C output in full: `ு暾原谅shotsIRログ Criteria.true`. Garbage output is expected, not a benchmark score. |
| Worst step/gradients | All tiny answers unfinished; no loss or gradient exists in this intervention. |
| Surprise | Short selected span contains `.\n\n`; explained: a merged tokenizer boundary token. Now disclosed in certificate, plan and report. |
| Missing evidence | Actual Qwen3.5 hybrid GPU identity/decode checks, mask on its tokenizer, semantic outputs, Jev scores and measured cost. Actual-model preflight is inside each walk before sweep generation. Historical cached C1 can also differ if its original batch/padding differed; investigate rather than weaken the assertion if this happens. |
| Alternatives | Pre-run subjective weights above: saturation45%, useful behavior25%, damage15%, implementation bug10%, unknown5%. Tiny identity/mask/decode checks argue against those implementation errors on Qwen3 only; scoring bias and generic-abrasiveness shortcuts remain untested until answer reads/Jev. No performance diagnosis yet. |
| Independent review | `code-review.md`: `Context-manager fit: resolved`, `Prefix slicing: resolved`, `Results endpoint: resolved`; actual-model path deferred to fail-fast preflight. Repeated-forward control added as requested. |
| Cheapest discriminator | C1 identity/decode preflight on actual 4B: a failure stops the sweep and identifies code/kernel trouble; a pass permits comparing the fixed-grid behavioral outputs. Health/Jev damage distinguish useful change from corruption. |
| Time/memory | Harness wall times: short smoke59s, engineered127s, ordinary smoke plus web build35s. CPU only; peak memory not measured. Cache reuse keeps reruns cheap. |

Updated preflight smoke passed for both methods in 33s: `outputs/logs/prompt-sweep-final-smoke.log` contains both `PROMPT_SCALE_CHECK_PASS` and `SMOKE_PASS` lines, including the repeated ordinary-forward assertion. Decision: proceed with the two bounded 4B dev sweeps. No extraction, no full cohort, no judging until generation finishes and a separate pull completes. Runtime checks do not yet say whether gain changes semantic instruction strength.
