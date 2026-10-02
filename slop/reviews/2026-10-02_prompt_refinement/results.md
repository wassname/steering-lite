# Prompt-gain refinement and denser random reference

PI/OpenAI, 2026-10-02. Branch: dev/prompt-gains-random-reference.

User: "ok do the scheudle, show me fixed plot"; "run more random interventiosn for better contours please"; "all future runs of plot have features we prototyping nw".

## Scope and completion

- [ ] Refine the short prompt's gain grid on the same 20 dev questions, seed 0, unchanged generator/personas/judge/damage cutoff. Preserve existing answer bytes; exact fresh gain-one identity must still pass.
- [ ] Extend the dev random reference from seeds 0–10 to 0–31. Existing full-100 reference remains 0–10. No simultaneous dev/full worker for a seed.
- [ ] Pull first, then Jev-judge separately. Check complete scenario coverage and old answer hashes.
- [ ] Regenerate normal dev and prompt-focused plots through results.py, with measured gains and actual random sample counts visible. Unmeasured spans must not appear as measured responses.
- [ ] Inspect PNGs and browser screenshots; fresh independent review; commit locally. No public push.

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

Pending execution. All future rendering changes belong in scripts/bsbench/results.py and web/, not a special saved-results renderer.
