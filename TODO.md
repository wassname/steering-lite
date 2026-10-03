# TODO

Open items from wassname, 2026-10-03 (quotes verbatim). Tick when done; link the evidence.

- [ ] Research: monotonic prompt-strength dials. > "please have a subagent (sol 6.1) do a search for ways people dial prompts up and down monotonically. pormpt prefix. look in sterrabiltiy for ideas too https://github.com/generative-computing/steerability/" Includes classifier-free guidance on logits as the known candidate. Output: `slop/research/2026-10-03_prompt_dials.md`.
- [/] Prompt gain grid: > "so for prompting can we onlyh ramp from 0 to 1 gain? with a min of 5%?" > "lets use log spacing from ~ to 1 then". Done in walk.py: quarter-octaves 2^-4.5..1 (19 gains). Running on Modal (full, then dev, sequentially: shared answer cache).
- [ ] Opposite of sycophancy that is not refusal. > "have you got any ideas for what's opposite to sycophancy but distinct from refusal?? maybe ask /oracle pnael for diverse opinions" Context: BullshitBench grades "Clear pushback / Partial challenge / Accepted nonsense" and removes refusals from the denominator.
- [ ] Maybe rename the -C persona (abrasive). > "we might also want to change abrasive -> something else? what does https://github.com/petergpt/bullshit-benchmark use" Answer: they score "clear pushback"; no opposite axis ("it does not measure how often models incorrectly reject valid questions").
- [ ] FIXME: count off-target answers as failure, not rejection. > "not responsing to question is a form of failure that I haden't considered! need to add ... this will mean bumping the eval version and redoing. metadata shoulsd have eval verison" Plan: Jev on_target per answer at every dose, weight premise effect by P(on target) or add it to the rubric; add `eval_version` to points.json and walk/judge metadata; re-judge.
- [ ] FIXME blind check: > "I showed the graph to a model blind and it totally misunderstood!!! not good we need to improve and iterate untill it's self evident to a blind agent." Show the PNG to fresh agents with no spec/code; iterate until they recover the message in the spec.
- [ ] Eval v2 rubric (with the FIXME above): separate incoherence (the stop) from side effects (the y axis); maybe report BullshitBench's three categories (clear pushback / partial challenge / accepted) alongside; flag prompt/role leaks (echoing the persona instruction, role tags, <think>) as a judged check, not only the regex diagnostic.
- [ ] Terms: x = behaviour change, y = side effects, stop = coherence/fluency. wassname: > "the generative paper uses the term fluency? should I ahve the terms I use to normalise? or emphasis it's a behaviur itnervention? should I inot seeprate incoherence/fluency from side effects?"
- [x] Faint dots: moot, lines now pass through every passing dose (whole sweep).
- [ ] Plot purpose agreed and written down: `slop/specs/20261003_bsbench_plot_purpose.md` (draft, needs wassname's edit).
- [ ] Plot: lines show the whole sweep, bare to last coherent dose (x), not only Pareto points. > "it's mean to describe the sweep from start to end, and the best tradeof is visually obvious as a pareto front"
- [ ] Random contours: zero-fill makes a weird edge at 0. > "the only problem is there's a weird effect at 0 because we will to zero. maybe we should not have done that?"
- [ ] query_steer user-turn walk not exhausted (still Jev-coherent at C=4096). Rerun with more rungs, about $1-2.
- [ ] Sound-premise control for user-turn -C (contrarian rejection?), about $1-2.
