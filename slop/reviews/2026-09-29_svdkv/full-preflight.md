# Full sink-split evaluation

PI/OpenAI. User approved the approximately $15 full evaluation before the naming discussion. Launch source: 05a83da, with unchanged estimators renamed in fdcf8e8. Original svdkv becomes sink_split; svdkv_resid becomes sink_split_resid.

Question: does the combined attention/residual method's dev advantage over mean_diff persist on the standard 100-question, three-seed Qwen3.5-4B protocol?

Choice: run both methods, seeds 0,1,2, cohort full, with at most 32 dose steps. Reuse existing full mean_diff and random controls. Jev judging is a separate step after generation/pull. Do not run dev and full simultaneously for the same method/seed: they share answer files.

Evidence before launch:
- `outputs/bsbench/results/dev/index.md`: sink_split_resid score +1.16 [90% CI +0.41,+1.76]; mean_diff +0.70 [+0.22,+1.41]; sink_split +0.52 [-0.03,+1.07]; random +0.05 [-0.18,+0.61]. These are selected dev doses, 20 questions; they do not establish a paired gain.
- `outputs/logs/method-renames-check.log`: 55 tests passed; SMOKE_PASS for the real tiny-model benchmark. Source AST comparison found no executable changes apart from names/string labels.
- `slop/reviews/method-naming/verification.log`: 29 renamed cached vector configs deserialize; saved benchmark statistics unchanged. `volume-migration.json`: all 1099 remote before/after hashes verified.
- Zero-dose KL on the target model was reported by the author over eight dev prompts: last-token mean 0.015, max 0.043 nats (source: sink_split.py module docstring). This is small in those prompts, not a guarantee of an identity intervention or a bound over full questions.

Options and predictions:
- Full evaluation (chosen): resolves whether the dev point-estimate advantage survives more questions and extraction seeds. No retuning.
- Repeat dev only: tests extraction variability but leaves question-selection uncertainty; not sufficient for the README full comparison.
- Centre the query split: changes the method; deferred because the target-model zero-dose check was small and the current comparison should hold the implementation fixed.

Working hypotheses, subjective planning weights: transferable composite improvement 45%; dev dose/question selection explains the apparent gain 30%; residual rescaling or zero-dose distortion explains it 15%; implementation/evaluation error 5%; unknown 5%. These are not result probabilities derived from a statistical model. Attention-only and existing mean_diff separate component behavior, but without a random-slot composite they cannot identify the source of every composite gain.

Expected discriminator: compare full headline scores and paired question-level differences, with dose selection handled consistently. Read actual answers at selected doses and near breakdown. Improvement accompanied only by increased rejection/refusal or damage does not establish better targeted steering. Full evaluation includes the dev questions, so it is not an independent held-out test; report that limitation.

Cost assumption: approximately $15 for six walks plus judging, approved earlier; wall time and actual charges remain to be measured. Smoke outputs have no scientific meaning. No optimizer, LR schedule, training loss or training gradients apply: these are extracted interventions and generation/evaluation runs.

SHOULD: each completed certificate has all 100 questions at both signs for the recorded dose steps, finite extracted state, and a confirmed health boundary. ELSE inspect incomplete generation/cache reuse before judging. SHOULD: Jev's content-keyed cache retains existing ratings; a rename alone should create no new judge requests.
