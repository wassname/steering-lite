# User-turn steering on BS-bench full (Qwen3.5-4B)

PI/OpenAI, 2026-10-02 (overnight, user AFK). Branch dev/prompt-gains-random-reference.

User request:
> yes try  user only variant of all, spend <$50. good night show me this user only graph in morning. small model. full bs bench. contoured random plus woith increased 10->50?
> and the dosed prompt one on same graph get them all done and hsow pls

Motivation (another agent's result, quoted by the user, not verified here): VJP steering applied only while the model reads the user's question reached effect −2.63 at damage .46 versus −1.75/.44 for steering everywhere.

## Design

- `src/steering_lite/positions.py`: `only_tokens(mask)` keeps each steering edit only at masked prefill positions. Cached decode steps (generated tokens) are unsteered; they see the steered prompt only through attention.
- Mask = tokens of the user message (`question + " Answer in 2 short sentences."`). Not the chat template, not the assistant header, not answer tokens.
- Same vector and same C0 as the steering-everywhere walk (`vectors/<method>_s0`, `calib/<method>_s0`), so C values are comparable. Results are named `<method>-user`.
- Walk: start at C0/2 (normal walks start at C0/8), stride 2 (third-octave), stop at 2 unhealthy rungs + 1 as usual, or gracefully at 18 rungs (≈25×C0). Prompt-only steering may stay mechanically healthy; the cap bounds cost. Jev mean damage ≤1.5 alone decides admissibility.
- Methods: every registered method except sink_split / sink_split_resid (extra attention slots are read by every query; no per-token form, the code refuses). One seed (s0), full 100 questions.
- Reference: `random-user`, random directions steered the same way, seeds 0–49 (budget permitting).
- Prompt sweeps (`prompting_scale`, `prompting_engineered_scale`) and plain prompts already act only on the prompt; run on full and shown on the same graph.
- Jev audit (new, separate request, existing ratings unchanged): at Pareto-best doses, P(on target) and P(fabricates). Probe on three hand-written answers (`audit_probe.py`): clean rejection on_target .99 / fabricates 0; fabricated rejection .61 / 1; off-target safety refusal 0 / 0.

## Checks before the run

- `tests/test_pipeline.py -k user_positions`: 6 passed (mean_diff, linear_act, value_gram, vjp_value, query_steer; sink_split raises). Empty mask = bare logits and greedy generation, tokens before the span unchanged, span steered.
- CPU smoke (`smoke-*.log`): `USER_POSITIONS_CHECK_PASS` for mean_diff and value_gram, vector cache hit.
- On Modal Qwen3.5-4B (pilot.log): `USER_POSITIONS_CHECK_PASS ... span_tokens=[43, 40, 47] prompt_tokens=[55, 52, 59]`.

## Cost plan (forecast; recorded before results)

Pilot `mean_diff-user` full s0: 1526 s Modal wall incl. container start, 18 rungs (hit the cap; −C broke at C=16–20, ≈19–24×C0, +C only `unfinished` at 20). At $1.95/h L40S ≈ $0.83 GPU-only. Jev ≈ 3600 aware ratings ≈ $0.17 per walk.

| Item | Walks | GPU est. | Jev est. |
|---|---:|---:|---:|
| pilot (done) | 1 | $0.83 | $0.17 |
| learned user-turn, 17 more | 17 | $14.1 | $2.9 |
| prompt sweeps full | 2 | ~$2 | ~$0.4 |
| random-user wave 1, seeds 0–23 | 24 | $19.9 | $4.1 |
| blind + audit | | | ~$1 |
| **total** | | **~$37** | **~$8.6** |

≈$46 forecast; random-user wave 2 (toward 50 directions) only if measured cost leaves room under $50.

## Results

(pending)
