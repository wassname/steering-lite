# Steering-lite BS-bench rewrite

Rewrite the benchmark from vjp-steering on the existing branch. Make a cheap first comparison of steering strength and precision on sycophancy, suitable for use and demo.

## User-visible result
On `rewrite/bsbench-vjp`, a clean cached Modal `just sweep` will compare steering-lite methods, VJP-cache and VJP-diff with prompting/random/mean-difference/PCA controls on 20 numbered Qwen3.5-4B BS-bench v2 questions, validate persona pairs, save on/off-target scores and blind change descriptions, produce matching reference-style HTML/PNG plots and maximum-coherent/optimal-dose tables, and test cheap method/model-specific RMS-KL calibration on about four new-data cases, for less than $50.

## Preferences and authority
- Initial GPU runs and judge calls together cost <$50; use cheaper subagents, not Astra. Continue with DeepSeek V4 Flash unless a cheaper suitable model is needed.
- Preserve the existing judge protocol; add a separate blind call. Dose-selection score: directed intended effect − 4 × off-target effect; retain the full Pareto frontier.
- This is experimental development, not a publication-proof study. No public posting or larger-model spending without separate approval.

## User voice
> can we add VJP-cache and VJP-diff

> yes please. also I wonder if we should uplift steering-lite to match vjp-steering so
>
> - **pairs** I would bring in minimal things in a clean way from  https://github.com/wassname/persona-steering-template-library, namely consider judge validation of the prefix and suffix? and perhaps it can be the same judge and reused from the eval? ok that's the steering personas
>   - but I would also like to use the judge (slow) or activation or J lens to validate if the person pairs correspond to the intended  
> - **calibration** I'm searching for a good way to calibrate so I can find the best dose cheaply and compare pareto optimal or max doses. A grid search is the most expensive. An iteratitive search such as an method of successive approximation (https://en.wikipedia.org/wiki/Iterative_method) is faster. But I'v found that (https://github.com/wassname/isokl_steering_calibration). I hope It can find the hyperparameter "target rmse(KL) [nats]" once per method/model, and it will allow quick calibration from then on. I would like to prove this... showing it predict's the max coherent dose on other datasets. (in any case it's better than what other papers and libraries do which is: no think about or talk about this at all, or do a super expensive grid search which I just can't afford).
> - **eval: faithfull** I found using a judge it usefull when the eval is not fully aligned with the steering, the judge can tell use "the effect is bluntness" but we aimed for honesty. another steering method might be more faithful. In fact the eval I used in jvp-steering seems good: it's fairly cheap (100 generations), fairly sensitive, has a nice graph showing the parto, and people understand sycophancy.
> - **platform** run on modal for speed, and still at low cost, and able to scale up from the initial Qwen 3.5 4B model to ~30B or 100+B
>
> so in the end I hope to redo steering-lite with a bullshit bench v2, judged eval, and my nice plot. Hopefully I can have a max dose and optimal dose table too. And compare to prompting as a baseline. Keeping the random zone, and mean mass diff, and PCA baseline.
>
> I migth use the 20 q dev versions first. ofc  the eval needs to be numbered. 
> In the end I hope to show the best steering method, ansering these questions
>
> which is the best (powerful and precise)?
> is it reliable? does it steer the concept we want fiathfully? does it scale to larger models? and I want a) html plot like vjp-steering (you need to view both png's and compare parity), b) corresponding max dose, and pareto optimal table
>
> the judge must also (this is a bew feature) 1) guess the concept being steered (ideally blind) this will tell us how faithfull it in
>
> need to be clean. we need to be able to run on modal with a `just sweep` where previuslly run ones are cached. we need to use https://github.com/wassname/vjp-steering as a base

> use cheaper models for subagents not astra

## Goals
1. [ ] goal: Rewrite the BS-bench v2 benchmark and compare VJP-cache/VJP-diff with the controls
   - Scope: reuse vjp-steering, preserve steering-lite variants including cache Gram, and include bare, prompting, random, mean difference and PCA; implement cache-value VJPs and reuse VJP-delta as VJP-diff, not a duplicate method.
   - Subtle failure mode: a parallel framework or a nonzero but wrong gradient looks like the requested rewrite.
   - Discriminator: the branch runs the real numbered benchmark; tiny-model extraction, actual cache-value derivatives, attachment, generation and save/load agree with independent checks, including Qwen3.5 hybrid attention.
   - Evidence:
2. [ ] goal: Validate persona pairs and report judged effects plus blind change descriptions
   - Scope: minimally reuse prefix/suffix and intended-behavior checks; retain paired base/steered on/off-target judgments and add target-blind descriptions with rough magnitudes.
   - Subtle failure mode: persona style substitutes for the intended concept, or target/method/dose metadata tells the blind judge what to find.
   - Discriminator: saved pair examples, full judge requests/responses and numbered outputs expose actual changes and disagreements; blind requests omit target metadata and allow no detectable change.
   - Evidence:
3. [ ] goal: Find useful doses cheaply and test method/model-specific RMS-KL calibration transfer
   - Scope: use iterative search rather than an exhaustive grid; find a useful RMS token-KL target in nats per method/model and test predicted doses on about four new-data cases, including other datasets.
   - Subtle failure mode: near-zero steering passes coherence, reused inputs masquerade as transfer, or a short calibration misses later breakdown.
   - Discriminator: saved search histories and generations compare predicted doses with nearby observed useful/coherent doses on new data, show intended effects and late failures, and distinguish a measured boundary from a search limit; diagnose bugs and iterate within budget.
   - Evidence:
4. [ ] goal: Run a cached Modal `just sweep` within the initial budget
   - Scope: Qwen3.5-4B, 20 numbered development questions; GPU work on Modal, CPU/API judging local; real-pipeline smoke before paid experiments.
   - Subtle failure mode: a rerun pays again, stale outputs survive a meaningful change, or missing billing hides overspending.
   - Discriminator: an unchanged rerun reuses completed work; changed inputs invalidate affected stages; saved manifests, costs and conservative unresolved-cost reservations keep the initial GPU/judge total below $50.
   - Evidence:
5. [ ] goal: Produce reference-style HTML/PNG Pareto plots and maximum-coherent/optimal-dose tables
   - Scope: preserve the random region and full frontier; report maximum coherent dose separately from the working 1:4 optimum, with links to numbered examples and judge disagreements.
   - Subtle failure mode: attractive plots hide rejected doses, differ from the tables, or imply a general winner from this small comparison.
   - Discriminator: shared measured points feed HTML, PNG and tables; inspect the reference and new PNGs and obtain a cheap independent visual review; explain which methods are powerful/precise here and which results remain uncertain.
   - Evidence:

## UAT / Verification
Open the report, compare the reference/new PNGs, inspect selected and failing numbered response pairs, and rerun `just sweep` from cache; trace plotted/table values to saved judgments, calibration histories and costs.

## Future work / out of scope
The full 100-question evaluation and ~30B/100+B experiments follow the initial interpretable comparison and separate spending approval; retain a scalable Modal entrypoint, but do not claim measured larger-model reliability now. No new concepts, trained steering objectives or unrelated J-lens framework.

## Log
### 2026-09-19 — planning resync — PI/OpenAI
- Planning only. Preserve existing work; do not launch workers, edit implementation or spend on experiments before Ready.
- Existing worktree: `/workspace/2026/lite/steering-lite-bsbench`, branch `rewrite/bsbench-vjp`, HEAD `820888ebb22a07dbf786a0bf8ea90b2c1cbe4a92`.
- Earlier plan: `slop/plans/20260919_bsbench-rewrite.md`. This file is the active approval draft. No scope expansion is proposed.
- Existing completed work: job 1704 audit committed at `7987a93`; target-aware judge port and separate blind-request schema committed at `820888e`; `slop/verification/20260919_judge-regression.log` reports 7 passed. Preserve these; do not restart them merely because this plan has open outcome checkboxes.
- Existing unfinished work: generation/data/cache modules, VJP-delta and VJP-cache, cache installation changes, test changes and dependency files are uncommitted. Inspected git status; leave them intact during planning.
- Latest VJP test log: `slop/verification/20260919_vjp-pipeline.log` reports `1 failed, 11 passed, 32 deselected`. Failure: `AdditiveValueCache._edit` treated an integer layer index as a tensor. A local fix and stronger hybrid test were written afterward, but not rerun. Do not report them as verified.
- Process inventory shows the judge, dependency and VJP test processes exited. The reconnaissance workflow has finished; its benchmark child needed a same-protocol retry. No new paid GPU or judge experiment has started.
- Grilling frontier is empty: prior answers settle outcome, evaluation principles, scope and spending. Remaining technical choices do not require another approval question.

## Interview
### Earlier grilling, recovered 2026-09-19
Source: `/home/code/.pi/agent/sessions/--home-code-dev--/2026-09-17T23-19-44-074Z_01a0b1ab-3ac9-7246-8341-22346f8a85fe.jsonl`. The following are verbatim answer excerpts; the source retains the complete questions and answers.

Scope:
> A comparison on sycophancy. As I said we want to bring over vjp-steering which does sycophancy via bs-bench and has a nice plot. Just use this instead of what we have. a rewrite. on a branch.

Blind judgment and research standard:
> I'm looking for disagreement actually (disprove not prove). I know this.

> But no one else is doing this, and I want to try this. Please don't ask for perfection when no other researcher is even addressing this in the first place.  We need to aim for doable not to widen the scope beyond the resourcesd we have. It's a first pass suitable for use and demo.

> Look at how it's already set up before asking me. iirc it sees base and steer.
>
> In current form we tell ti the target, and ask for on and off target.
> But in the suggested additonal form it would not be given the target and be asked to describe the change with words and magnitudes perhaps?

Calibration:
> what do you mean by learning? oh I see you mean does it generalise. Well yes that's why I said I want to know if it predict in new data. But this is quick cheap dev at the moment so just see if it worked for N=4 or similar.

> Claim? This is experinmetnaton to try and get it working. It's a experimental hypoithesis not a academic paper. You are taking the wrong approach.

> this was never the aim. The aim is to cheaply and roughly calibrate. I have qualitativly observed it kind of works but differs between methods. We can't use 1 nat for all methods. Need to find the right mark for each method. And cheap quickly if it transfers, if we disprove it then it doesn't, if it seems to, that's initial weak evidence and usefull to go forward with to move testing.

Dose selection:
> correct. eyeballing the vjp-steeing graph where the optimal vjp-delta is -1.5 on axis and  0.25 off axis. we can use a 1:4. Working number.

Budget and staged scaling:
> perfect yes, spend <$50 on this initial set. We have costs from the other project, that shoudl be fine.

Worker preference:
> use cheaper models for subagents not astra

Rejected options:
- ~~A publication-validation study or proof of general faithfulness~~ — wassname rejected this scope expansion; seek useful disagreements and weak initial evidence.
- ~~Another parallel benchmark~~ — wassname requested a clean rewrite using vjp-steering on a branch.
- ~~One fixed 1-nat target for every method~~ — wassname says the suitable target differs by method.

### 2026-09-19 — current planning request
> mk plan

No unanswered protected choice. Technical unknowns remain: which methods work, which RMS-KL targets transfer, what dose boundaries occur, and exact measured costs. These are experiment outcomes, not questions for the user to answer in advance.

## Learnings
- Reference code: `/workspace/2026/jspace/j-steer_pub`, `dev3`, commit `efcd84804a400360e383fef7ac24749bc5855bbf`, remote `https://github.com/wassname/vjp-steering.git`; reuse its BS-bench generation/judge/export/results path, not unrelated experiments.
- Reference judge supplies the target and compares bare/steered in AB and BA order. Blind discovery is an additional call, not a replacement rubric. Judge agreement alone is not evidence of faithful steering.
- Existing `vjp_delta` already computes a positive-minus-negative class-mean VJP difference. Swapping both classes leaves the raw difference unchanged; the global orientation heuristic supplies its sign. Do not create a duplicate method or use raw label-swap antisymmetry as a false test.
- VJP-cache must differentiate the later hidden-state target through actual earlier cache values; a nonzero projection-hook gradient alone does not establish that path. Only full-attention value caches are intervention targets in hybrid Qwen3.5.
- Exact calibration repository fetched successfully: `https://github.com/wassname/isokl_steering_calibration`, commit `27bfbcdd2df2d2e7103dfbc64aae3936b3415812`. Its 2026-09-09 README update says: “I observed that it differs by method. So I hope to show that this hyperparameter can be meta-calibrated once per method (and likely per model)”. This is the author's hypothesis/observation, not validated transfer evidence. Its code still defaults to `target_stat="kl_p95"`; steering-lite already has RMS-KL and an iterative solver.
- Read `https://en.wikipedia.org/wiki/Iterative_method`: successive approximations are the intended computational approach, not a reason to add a new optimization framework.
- Persona source `/workspace/2026/weight-steering-repos/persona-steering-template-library` has the requested remote. Its README says it rejects prompts where “refusal, answer length, style, or copied persona labels explain the difference better than the intended behavior.” Reuse a small relevant part, not its full screening framework.
- Source cost records suggest inexpensive Flash judging, but historical averages are estimates, not a cap or current billing. Count failures/retries and keep unresolved spending reserved.
- Package instructions require real functional tests, one file per method, decorator registration, fail-fast behavior, `einops.einsum` and shape annotations. Referenced optional sibling `/workspace/2026/lite/lora-lite/AGENTS.md` was unavailable: `ENOENT: no such file or directory`.

## Papercuts - problems, gotchas, suggestions
- A scout's broad recursive grep blocked in kernel I/O for over 13 minutes. Use direct known source paths; do not rescan all of `/workspace/2026/lite`.
- Some scout text contains incorrect source-hash transcription and an incorrect raw label-swap test suggestion. Prefer the verified git hashes and source equations above.
- Check optional-Transformers imports in the unfinished VJP-cache module; importing a new variant must not accidentally break the package's declared installation modes.

## Appendix (context, not approved)
- Exact extraction counts, random seeds, transfer datasets, dose-search bounds and judge token limits remain reversible implementation choices within the settled initial scope and budget. Record their actual values with results.
- Distinguish the RMS-KL calibration hyperparameter from an extracted steering vector. Do not tune a target on a transfer case and then call that case a prediction.
- The old cache-Gram result is a documented negative at one operating point. Preserve its audit; do not turn it into a successful leaderboard entry or treat it as proof that all cache steering fails.
- Resume by inspecting the existing diff and verification logs, fixing/testing unfinished methods and integration, then running a bounded real smoke before Modal spending. Do not re-create the branch or overwrite uncommitted work.

-- PI/OpenAI
