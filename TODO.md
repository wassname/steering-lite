# TODO

Open items from wassname, 2026-10-03 (quotes verbatim). Tick when done; link the evidence.

- [ ] Research: monotonic prompt-strength dials. > "please have a subagent (sol 6.1) do a search for ways people dial prompts up and down monotonically. pormpt prefix. look in sterrabiltiy for ideas too https://github.com/generative-computing/steerability/" Includes classifier-free guidance on logits as the known candidate. Output: `slop/research/2026-10-03_prompt_dials.md`.
- [ ] Prompt gain grid: > "so for prompting can we onlyh ramp from 0 to 1 gain? with a min of 5%?" Proposal: gains 0.05..1, dense in 0.05..0.12 where the switch is (full +C: 0.0625 -> +1.05, 0.094 -> +3.33).
- [ ] Opposite of sycophancy that is not refusal. > "have you got any ideas for what's opposite to sycophancy but distinct from refusal?? maybe ask /oracle pnael for diverse opinions" Context: BullshitBench grades "Clear pushback / Partial challenge / Accepted nonsense" and removes refusals from the denominator.
- [ ] Maybe rename the -C persona (abrasive). > "we might also want to change abrasive -> something else? what does https://github.com/petergpt/bullshit-benchmark use" Answer: they score "clear pushback"; no opposite axis ("it does not measure how often models incorrectly reject valid questions").
- [ ] FIXME: count off-target answers as failure, not rejection. > "not responsing to question is a form of failure that I haden't considered! need to add ... this will mean bumping the eval version and redoing. metadata shoulsd have eval verison" Plan: Jev on_target per answer at every dose, weight premise effect by P(on target) or add it to the rubric; add `eval_version` to points.json and walk/judge metadata; re-judge.
- [ ] Plot purpose agreed and written down: `slop/specs/20261003_bsbench_plot_purpose.md` (draft, needs wassname's edit).
- [ ] Plot: lines show the whole sweep, bare to last coherent dose (x), not only Pareto points. > "it's mean to describe the sweep from start to end, and the best tradeof is visually obvious as a pareto front"
- [ ] Random contours: zero-fill makes a weird edge at 0. > "the only problem is there's a weird effect at 0 because we will to zero. maybe we should not have done that?"
- [ ] query_steer user-turn walk not exhausted (still Jev-coherent at C=4096). Rerun with more rungs, about $1-2.
- [ ] Sound-premise control for user-turn -C (contrarian rejection?), about $1-2.
