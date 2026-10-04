# TODO

Open items from wassname (quotes verbatim). Tick when done; link the evidence. Work top to bottom.

## Now: cost and design fixes (wassname 2026-10-04)

> "rm threshold and fix plt"
> "you messed that one up. or else everything need to be batched to fill up gpu FIXME"
> "well why don't you fix? use an image with it in? I'm sure there is one? FIXM"
> "while validating we don't need to do all, baseline, random, and best will do"
> "I'm leaning towards "just bs bench" just pushback vs sycoiphancy and close to that"

- [x] Remove the 5 pp false-pushback threshold (added without approval); regenerate plots.
- [x] Remove prompt gain sweeps (> "we need to remove prompt sweep"; broken below 5% and above ~8% gain).
- [ ] FIXME GPU: 4B bf16 on an L40S ($1.95/h). Use L4 ($0.80/h) or fill the GPU with bigger batches (64-128).
- [x] FIXME image (checked: only causal-conv1d missing; with it, 1–5% faster, not worth torch 2.10). transformers says "The fast path is not available because one of the required library is not installed" (likely causal-conv1d); build the Modal image with it and confirm the warning is gone.
- [ ] max_new_tokens 512 -> 160 (healthy answers: median 47-57 words, max 93 words, about 125 tokens).
- [ ] Twins only at each method's chosen dose, not every dose (halves generation).
- [ ] Validation runs: baseline (mean_diff), random, best (vjp_resid) only.
- [x] Oracle: how to cut Modal cost (`slop/research/2026-10-04_modal_cost/`), and benchmark (journal 2026-10-04): one batch of 200 on A10G or L40S, cap 192, is 3–4× cheaper per answer.
- [ ] tyro config: one subconfig per model size (model, batch size, max_new_tokens, Modal GPU); validate each fills the GPU on one run before branching out.
- [ ] Decide with wassname: plain BS-bench axis (pushback vs sycophancy), twins as a check only.
- [ ] Twins were read by hand and checked by GPT-6.1-Sol, not verified with web search; verify with search.

## Earlier: eval v2, the "accurate" axis and the control set (approved 2026-10-03)

> "yes switch, and add control set. looking forward to new graph. spend naother $50 or wharever you need."
> "after the first run make sure to check generations manually as per ml-debug ko"

1. [x] Pick the opposite pole. `accurate` is the only persona that raised pushback on nonsense without raising false pushback on sound questions (net +7 [+3, +13]; `pushes back` -11). Evidence: `slop/research/persona_poles/results.md`. Oracle panel: `slop/research/2026-10-03_oracle_opposite_of_sycophancy.md`.
2. [x] Control set: 100 sound-premise twins, `data/bsbench/sound_twins_v1.jsonl` (writer Opus 5.5, checker GPT-6.1-Sol, all 100 read by hand, 8 hand-written or fixed).
3. [x] Eval v2 code, no spend (commits 0519007, 18862a7, e1396c9):
   - axis `sycophantic` vs `accurate`; engineered prompt for `accurate` regenerated with the same gpt-4o recipe;
   - axis and `eval_version` in the generation key, so v2 writes to a new output dir and never reuses abrasive-axis vectors or answers;
   - every tested dose also answers the 100 twins; judge them with `judge.control_request` (false pushback);
   - FIXME off-target as failure: Jev audit (`on_target`) at every dose; per-answer effect = P(on target) x premise change (an off-target "rejection" earns ~0);
   - report false pushback next to pushback (table + one extra chart); display-only, no new filter threshold;
   - remove the temporary `before_reversal` display rule if on-target weighting removes the corda swerve;
   - smoke tests (`just check`).
3b. [x] Overnight decisions (wassname 2026-10-03: "don't know, gtg, please make use of the night and time"). Written by PI/OpenAI BEFORE seeing the pole-screen results:
   - false pushback becomes an admissibility limit like the damage cap: a dose counts only if Jev false pushback on twins rises <= 5 pp over bare (both sides). Pilot showed the score picking vjp_resid user -C at C=0.79 with +36 pp. wassname can switch to a net score later; every point keeps its FP so nothing is lost.
   - pole rule: from the mean_diff screen (accurate, candid, skeptical, abrasive), pick the pole with the largest mean_diff -C pushback at a dose with damage <= 1.5 and FP <= 5 pp. Tie or all ~0: keep accurate (prompt-screen winner) and report mean_diff failing.
   - if the pole changes, the engineered prompt is regenerated with the same gpt-4o recipe and the prompt baselines are rerun on it.
4. [x] First run, small: mean_diff + vjp_resid, user turn and everywhere, 4B full. Then check generations by hand per ml-debug (read answers at low / best / last dose for both sets, quote them), before anything else runs. Evidence: `slop/reviews/2026-10-03_eval_v2/pilot_read.md` (mean_diff -C with accurate goes the wrong way; vjp_resid user -C picks a contrarian dose).
5. [x] Main run (budget about $50; axis now sycophantic vs skeptical, see 3b and `slop/reviews/2026-10-03_eval_v2/pole_screen.md`): mean_diff, vjp_resid, vjp_value, sspace_scale, corda_pca, chars, linear_act; user turn and everywhere; random reference; prompt + engineered prompt + both gain sweeps. 4B full.
6. [x] Pull, Jev judge, build reports, browser UAT, look at the PNGs. `outputs/bsbench/results/v2-everywhere/`, `v2-user/`; manual read `slop/reviews/2026-10-03_eval_v2/main_read.md`.
7. [x] Blind check (Fable 5.1, Sol; `slop/reviews/2026-10-04_blind_plot/`): both recovered the main reading; fixed missing prompt stars and units. Second round not done: show the new plot to fresh agents with no spec or code; iterate until they recover the message in `slop/specs/20261003_bsbench_plot_purpose.md`.
8. [/] Fresh-eyes review (Sol, `slop/reviews/2026-10-04_fresh_eyes/`), README eval v2 section (8a5b9dd), journal entry, commit.
9. [x] Send `mpc` and `lucid24` the judge code path, how to run it on a generations.jsonl with a BASE column, and the axis name.

## Later

- [ ] corda_pca flips sign with the skeptical pole (v1 abrasive was right way). Check |cos(top PC, mean diff)| per layer; if small, the sign rule is fragile.
- [ ] More seeds for the top methods (one seed per method in eval v2) and a second random batch; the 5 pp cap decides most -C doses, so seed noise at the cap matters.
- [ ] wassname to decide: false pushback as a cap (current, 5 pp) or a net score.

- [ ] Research: monotonic prompt-strength dials (classifier-free guidance on logits is the known candidate). > "please have a subagent (sol 6.1) do a search for ways people dial prompts up and down monotonically" Output: `slop/research/2026-10-03_prompt_dials.md`. Sol run queued in the oracle chain.
- [ ] Prompt gain dial is set by the first RMSNorm: instruction-token embedding RMS is about 0.013, eps 1e-6, so RMSNorm cancels the gain above about sqrt(eps)/0.013 = 0.077 (fraction of normal layer-0 input: g=0.0625 0.63, 0.088 0.75, 0.125 0.85). A real dial would scale after the first norm, or use CFG.
- [ ] Eval v2 follow-ups: separate incoherence (where a line stops) from side effects (y axis) with small yes/no Jev checks (makes sense, loops, role or prompt leak, off topic); terms on the plot. > "should I inot seeprate incoherence/fluency from side effects?"
- [ ] query_steer user-turn walk not exhausted (still Jev-coherent at C=4096).
- [ ] Ramp variant of user-turn steering: answer tokens ramp 0 to 1 over 32 tokens (wassname's idea, `~/.agents/skills/steering-concepts/references/token_position_steering.md`).

## Done

- [x] Plot spec agreed: `slop/specs/20261003_bsbench_plot_purpose.md`; lines show the whole sweep pinned at the x; rings and faint dots removed; random bands not zero-filled (commit a087934).
- [x] Prompt gain grid 2^-4.5..1 (log spaced), full + dev walked and judged.
- [x] BullshitBench rubric checked: clear pushback / partial / accepted; v2 has no controls.
