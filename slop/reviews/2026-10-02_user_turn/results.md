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

### Budget correction (23:35 AWST)

Measured walk times were higher than the pilot: 27 COMPLETE walks summed 55,822 s ($30.2 GPU-only at $1.95/h); random-user walks average ≈1,900 s (≈$1.03), sspace-family learned walks 3,000–4,500 s. Forecast with all 24 random seeds ≈$55 including Jev, over the <$50 cap. I stopped the random wave (`modal app stop`; queued seeds 16–23 never ran, s16 has one orphan RUNNING rung and is excluded) and resumed seeds 7–15 from their cached rungs (`random-walks-resume.log`). Random-user reference: **16 directions (s0–s15)**, not 50. Raising it toward 50 would cost ≈$35 more.

Note: Modal reuses warm containers for queued inputs, so stopping only newly started containers would not have stopped queued seeds.

### Spend (GPU-only proxy, not invoice)

| Item | Seconds | $ |
|---|---:|---:|
| certificates `timing.total_s`: 18 learned-user, 16 random-user (+ s16 orphan), 2 prompt sweeps | 64,972 | 35.19 |
| random s7–s15 work before the stop (certificates restart timing on resume) | ≈10,400 | ≈5.6 |
| container start / snapshot (≈50 containers × ≈50 s) + failed first prompt attempt | ≈2,800 | ≈1.5 |
| **GPU subtotal** | ≈78,000 | **≈42.3** |
| Jev pass 1: 82,751 aware + 5,712 blind | | 4.07 |
| Jev pass 2: 29,139 aware + 997 blind + 18,701 audit | | 1.95 |
| Jev audit for compare.py (588 cells) | | 0.02 |
| **Jev subtotal** | | **6.04** |
| **Total** | | **≈48.3** |

Excludes Modal CPU/memory and invoice reconciliation. No further generation.

## Results

Report: `outputs/bsbench/results/user-full/` (plot.png, index.md, index.html). Paired comparison: `comparison.md` (`compare.py`).

### Same vector, user turn vs everywhere (seed 0, 100 questions)

Score = weaker side's best admissible (on-axis − off-axis). Δ CI resamples questions only (paired, dose selection redone), one seed, so it omits seed-to-seed variation; dose selection is in-sample.

| Method | User turn | Everywhere s0 | Δ [90% CI] | −C on/off user | −C on/off everywhere |
|---|---:|---:|---|---|---|
| vjp_resid | **+2.47** | +0.75 | **+1.72 [+0.94, +2.02]** | +3.22/0.53 | +1.03/0.29 |
| sspace_scale | +1.59 | −0.14 | +1.73 [+0.96, +2.22] | +3.42/0.79 | −0.02/0.12 |
| vjp_value | +1.50 | +1.15 | +0.35 [+0.00, +0.68] | +2.10/0.60 | +1.59/0.44 |
| corda_pca | +1.13 | +0.26 | +0.87 [+0.33, +1.13] | +1.47/0.22 | +0.42/0.16 |
| mean_diff | +0.83 | +0.38 | +0.46 [−0.33, +0.69] | +1.79/0.96 | +0.57/0.19 |
| random (16 vs 11 directions, unpaired) | +0.45 | −0.07 | — | +1.54/1.09 | +0.05/0.12 |
| linear_act | +0.08 | +0.74 | −0.66 [−1.11, −0.42] | +0.27/0.19 | +1.09/0.35 |
| chars | +0.02 | +0.84 | −0.82 [−1.11, −0.37] | +0.49/0.47 | +1.39/0.55 |

Full 19-row table in `comparison.md`. User-turn helps 4 methods clearly (vjp_resid, sspace_scale, corda_pca, vjp_value marginal) and hurts several (linear_act, chars, spherical, topk_clusters, value_gram, pca).

### Are the −C "wins" real rejections? (Jev audit)

P(on target)/P(fabricates) at the Pareto-best −C dose; bare answers .89/.52.

| Method | User turn | Everywhere |
|---|---|---|
| vjp_resid | .87/.20 | .91/.36 |
| sspace_scale | .74/.18 | .88/.54 |
| vjp_value | .84/.25 | .88/.33 |
| corda_pca | .92/.29 | .90/.45 |
| mean_diff | **.54**/.26 | .91/.41 |
| random-user | **.50**/.30 | .89/.49 |

Read: vjp_resid user-turn −C answers stay on target and fabricate less than bare (examples-vjp_resid-neg.md: "There is no 'Krantz-Morrison framework' ..."). mean_diff and random user-turn −C "wins" are about half off-target (examples-mean_diff-neg.md: "the SaaS target does not exist", "you cannot use the provided text"), so their Jev premise score overstates them. Jev's premise rubric counts these as rejections; the audit is a separate diagnostic, not a filter.

### Coverage of the dose range

Every user-turn walk except query_steer reached Jev-inadmissible doses (last-rung mean damage 2.5–3.9) although most stopped at the 18-rung cap without a mechanical breakdown. query_steer stayed admissible at C=4096 with small effects: under-tested.

### Prompt sweeps on full

Gain-one identity passed 400/400 on identical batches. Same prompts in a different batch layout or process change 35–56 of 100 greedy answer texts (`cached_C1_mismatches`, `historical_mismatches` in prompt-walks.log), so every cached answer is one draw of a batch-sensitive process. Short prompt × gain scores +0.32 [+0.09, +0.82]; engineered −0.12.

### Follow-up checks after the independent review ($0)

Review: `review.md` (Anthropic, fresh context): no code or data blockers; asked for a per-seed and a held-out dose check. `split_half.md`:

- Seed 0 everywhere is typical: vjp_resid everywhere s0/s1/s2 = +0.75/+0.66/+0.66, so Δ is not inflated by a weak baseline seed.
- Held-out dose selection (pick dose on 50 questions, score on the other 50, 200 splits): vjp_resid Δ median +1.74 [+1.14, +2.21], sspace_scale +1.88, corda_pca +0.72, vjp_value +0.32 [−0.19, +0.74], chars −1.03. In-sample selection is not driving the headline.
- `+inf` upper ends in split_half.md: the everywhere side had no dose under the damage cap on the held-out half.

Open points from the review, not fixed tonight:

- Contrarianism: −C rejections may partly reject the question regardless of content (one answer rejects for an invented flaw, "Net is not an IDE theme"). BS-bench has no sound-premise questions, so the false-rejection rate is untested. Cheapest test: vjp_resid-user −C at C=1 and everywhere at its best dose on 20–50 sound-premise questions, ≈$1–2.
- Random-user reference is right-skewed (directions of either sign mostly push toward accepting); random's −C score row pools only 3 of 16 directions still admissible at C=4.
- The audit probe is a smoke test on one question, not a validation; its on_target and fabricates answers are entangled (fabricated rejection .61 on target).
- Plot caption should say it shows the 5 best-scoring of 21 methods; engineered-prompt × gain +C best dose is C=0 (instruction zeroed).
- Full benchmark report rebuilt with the new code: `full-regression.log` FULL_BENCHMARK_REGRESSION_PASS (all existing scores, selections and points identical; 80 prompt-sweep points added; CI endpoints of 14 methods shift by ≤0.039 because new methods enter the shared bootstrap RNG order). README images not updated.

### Caveats

- One seed per user-turn method; CIs resample questions only.
- In-sample dose selection (same 100 questions pick and score the dose).
- Random-user reference has 16 directions, not 50 (budget).
- User-turn walks start at C0/2 (normal walks C0/8) and cap at 18 rungs; doses are on the same grid.
- The other agent's report motivated this; its numbers were not reproduced here, only the direction (vjp user-turn much further left at similar damage).

-- PI/OpenAI
