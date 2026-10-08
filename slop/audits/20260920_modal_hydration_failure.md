# Audit: first real BS-bench sweep stopped before Modal dispatch

— PI/OpenAI, 2026-09-20

Target: `just sweep '--run --backend real' 'Qwen/Qwen3.5-4B' 'outputs/bsbench-v2'` at commit `f4af290` on `rewrite/bsbench-vjp`.

Provenance: complete 99-line primary log at `slop/verification/20260919T225817Z_real-sweep-first-run.log`; dry preflight at `slop/verification/20260919T225626Z_real-sweep-dry-preflight.log`; append-only ledger at `outputs/bsbench-v2/costs.jsonl`. Modal Python package: `1.5.5`.

Intended change: no algorithm change; execute the reviewed eight-method production graph. Resolve condition: complete the real sweep, make an identical cached rerun with no remote work, then render the report. Before running, the expected first observable was a hydrated Modal app and one completed bare generation stage. An identity/budget mismatch or unresolved remote outcome was specified to stop the run.

## Stage table

| stage | expected | observed | expected? | clues | missing metric | consequence |
|---|---|---|---|---|---|---|
| Credential check | Scoped OpenRouter key present without printing it | Presence check returned true | yes | `slop/verification/20260919T224920Z_real-sweep-credential-check.log:1-2` | Key validity was not exercised | Judge path remains untested |
| Dry preflight | 8 methods, 20 questions, 14 GPU stages, 904 aware + 904 blind + 12 persona calls, upper `<$50` | Exact identity; `$38.9767768` including prior `$2` | yes | `slop/verification/20260919T225626Z_real-sweep-dry-preflight.log:35` | None for planning identity | Paid dispatch was authorized |
| Production cache lookup | First bare generation misses | `cache miss generation 5a27bc2f0335` | yes | `...real-sweep-first-run.log:36-37` | None | Reservation and callback executed |
| Reservation | Reserve `$0.884346` before remote work | One reservation written | yes | `outputs/bsbench-v2/costs.jsonl:1` | Provider call ID absent | Accounting correctly became conservative |
| Modal app hydration | Function handle is runnable from the plain sweep process | `Function has not been hydrated ... App ... is not running` | no | `...real-sweep-first-run.log:77-96` | No Modal function-call ID | First production stage did not start |
| Bare generation | 20 Qwen3.5 answers and health records | Missing | no | run stops at line 96 | Answers, health, duration, GPU memory | No model/effect result is interpretable |
| Candidate extraction/search | Six vector methods produce vectors and candidate histories | Missing | no | earlier failure | All candidate metrics | Calibration is untested |
| Aware/blind/persona judgment | Saved complete requests/responses | Missing | no | earlier failure | All judge outputs and costs | Persona/faithfulness goals remain open |
| RMS-KL transfer/final generation | Fit per-method target and test four disjoint cases | Missing | no | earlier failure | Targets, predictions, final answers | Transfer goal remains open |
| Cache rerun/report | Zero-call rerun and real HTML/PNG | Missing | no | first command failed | Cache reuse and real plots | Goals 4–5 remain open |
| Artifact persistence | No partial result should masquerade as completion | No production-cache file or run summary; ledger preserves failure | yes | ledger lines 1–2; worker handover | Provider billing receipt | Safe but blocks retry |

## Chronological evidence

### Preflight matched the approved experiment

The dry artifact is the program's own resolved manifest, not a later summary:

> `"conditions": ["bare", "prompting", "random", "mean_diff", "pca", "kv_cache_gram", "vjp_delta", "vjp_cache"]` ... `"gpu_stages": 14` ... `"requests": {"blind": 904, "persona_validation": 12, "target_aware": 904}` ... `"total_upper_usd": 38.976776799999996`

Source: `slop/verification/20260919T225626Z_real-sweep-dry-preflight.log:35`. Epistemic context: machine-emitted resolved dry manifest from the same entrypoint immediately before the attempt.

This almost certainly establishes that the intended model/method/question/budget identity was selected; it does not establish that the live callback could run.

### The failure occurred before model execution

The first production-stage transition is visible with surrounding lines:

> `2026-09-20 06:58:25.055 | INFO | steering_lite.data.personas:load_suffixes:47 - Loaded 200 branching suffixes`
>
> `2026-09-20 06:58:25.057 | INFO | steering_lite.benchmark.cache:cached:37 - cache miss generation 5a27bc2f0335`
>
> `Traceback (most recent call last):`

Source: `slop/verification/20260919T225817Z_real-sweep-first-run.log:36-38`. Epistemic context: complete local stdout/stderr from the real command.

The call stack then terminates at client hydration:

> `File "/workspace/2026/lite/steering-lite-bsbench/scripts/run_bsbench_modal.py", line 310, in call`
>
> `return run_stage.remote(`
>
> `File ".../modal/_object.py", line 203, in _validate_is_hydrated`
>
> `modal.exception.ExecutionError: Function has not been hydrated with the metadata it needs to run on Modal, because the App it is defined on is not running.`
>
> `error: recipe 'sweep' failed on line 16 with exit code 1`

Source: `slop/verification/20260919T225817Z_real-sweep-first-run.log:77-99`. Epistemic context: Modal 1.5.5 client traceback; no application-generated remote line appears.

The implementation explains the message: `remote_stage_call` calls the locally declared handle directly:

> `def call(*, stage, method, config, prompts):`
>
> `    gate.require()`
>
> `    return run_stage.remote(`

Source: `scripts/run_bsbench_modal.py:308-310`. Epistemic context: exact executed commit `f4af290`.

No surrounding `app.run()` exists, and no deployed function is looked up. This makes the binding bug highly likely.

### Provider-side corroboration and deterministic reproduction

Eleven minutes after the failure, the authenticated Modal CLI returned:

> `modal_version=1.5.5`
>
> `active_or_recent_apps=[]`

Source: `slop/verification/20260920_modal-postfailure-app-list.log:2-3`. Epistemic context: provider CLI status query under profile `wassname`; an empty recent-app list is corroboration, not a billing invoice.

A minimal app handle called outside `app.run()` reproduced twice:

> `same-config ExecutionError Function has not been hydrated with the metadata it needs to run on Modal, because the App it is defined on is not running.`
>
> `irrelevant-config-change ExecutionError Function has not been hydrated with the metadata it needs to run on Modal, because the App it is defined on is not running.`

Source: `slop/verification/20260920_modal-hydration-minimal-repro.log:1-2`. Epistemic context: local Modal 1.5.5 reproduction; it isolates hydration from model/data/seed.

A direct import of the full Modal module did not finish within 120 seconds and produced no output (`slop/verification/20260920_modal-hydration-local-repro.log` is empty). That timeout is not evidence about remote execution and is not used in the diagnosis.

### Durable accounting state

The ledger did not erase uncertainty:

> `{"event": "reserved", "kind": "modal-generation-bare", "upper_usd": 0.8843460000000001, ... "id": "f6291f..."}`
>
> `{"event": "unresolved", "reservation": "f6291f...", "reason": "dispatch_or_validation_failure"}`

Source: `outputs/bsbench-v2/costs.jsonl:1-2`. Epistemic context: append-only local spending ledger.

`production_stage` reserves before invoking the backend and marks every exception unresolved:

> `reservation = reserve(ledger, f"modal-{stage}-{method}", config["upper_usd"], limit_usd=50.0)`
>
> `try:`
>
> `    result = backend.gpu(...)`
>
> `...`
>
> `except Exception:`
>
> `    mark_unresolved(ledger, reservation, "dispatch_or_validation_failure")`

Source: `src/steering_lite/benchmark/production.py:34-44`. Epistemic context: exact executed code.

This is conservative and correctly prevents a blind retry. The evidence strongly supports reconciling this reservation to `$0` with an append-only receipt/audit record, not deleting it.

## ML-debug form

| row | answer |
|---|---|
| log length; config as logged | Complete real log: 99 lines. Command/model/output at lines 1–3; dry resolved manifest at `...dry-preflight.log:35`. |
| each `SHOULD:` then observed | No `SHOULD:` line exists in this execution log. The pre-recorded execution condition was that identity/budget match and unresolved outcomes stop; both were observed. |
| numeric scales/nulls | Budget upper `$38.9767768` vs hard limit `$50`; first-stage reservation `$0.884346`; remote completions `0`; production-cache files `0`; run summaries `0`. Scientific metrics have no null/baseline value because model execution never began. |
| init demo | Missing because hydration failed before container/model init. |
| dummy/baseline | Bare was intended as baseline but produced no answer. |
| baseline on val/held-out | Missing. |
| schedule/lr | Not applicable; no optimization or model stage ran. |
| one full sample | Input prompts are identified by the dry manifest, but no remote-consumed sample/output/trace exists. This absence blocks scientific interpretation. |
| worst step metrics/grad | Not applicable; failure is infrastructure before tensor work. |
| surprising lines | `active_or_recent_apps=[]` corroborates the local traceback. Explained: no app was started. Full-module local import timing out is unresolved but not causal to the recorded failure. |
| missing trust evidence | Provider billing/FunctionCall receipt for the exact window; one successful single-method Modal callback; full raw answer/health records. |
| diagnoses | H1–H4 below. |
| fresh subagent | Independent Kimi K3 reviewer: “`remote_stage_call` ... calls `run_stage.remote()` on a Function whose `app = modal.App(...)` is defined at import time but never run and never deployed”; it estimated `~0.99` for this cause and `~0.97` that no remote function started. It also warned that strict OpenRouter JSON-schema support is a plausible next failure. This was a fresh read-only review with no preferred diagnosis supplied. |
| cheapest separating test | Provider app list + minimal local hydration reproduction separate client binding from model/data errors; both matched client binding. Next, a single bare-only run after binding should create exactly one Modal FunctionCall. |
| wall-clock/GPU | Real command: 9 seconds (`22:58:17`–`22:58:26`); remote GPU memory and stage duration missing because no function ran. |

## Hypotheses

### H1 [bug | Almost Certain | 97–99%]

- **Mechanism:** the sweep imports an app-bound `modal.Function` and calls `.remote()` without `app.run()` or a hydrated deployed handle.
- **Evidence:** `modal.exception.ExecutionError: Function has not been hydrated ... App ... is not running` (`...real-sweep-first-run.log:87-96`), plus `run_stage.remote` at `scripts/run_bsbench_modal.py:310` and the two-line minimal reproduction.
- **Contrary evidence:** the earlier phase-6 smoke ran through Modal, but it used Modal's entrypoint lifecycle; it does not exercise this plain-process callback.
- **Discriminating test:** bind through one explicit `app.run()` lifecycle or deploy/from-name; a bare-only run should produce exactly one app/function call. If the same hydration error remains, module/app identity differs from this hypothesis.
- **Fix/action:** run the canonical sweep inside one explicit app lifecycle, or deploy once and resolve a hydrated function by name. Prefer one lifecycle for the whole command so 14 stages do not create 14 apps.
- **Interpretability:** no scientific result; the infrastructure failure itself is interpretable.

### H2 [measurement | Highly Likely | 80–90%]

- **Mechanism:** the unresolved `$0.884346` is accounting uncertainty, not actual spend; local hydration failed before dispatch.
- **Evidence:** traceback ends in `_validate_is_hydrated` (`...log:87-96`) and Modal CLI reports `active_or_recent_apps=[]` (`...app-list.log:3`).
- **Contrary evidence:** app-list is not a billing invoice, and the local ledger has no provider call ID.
- **Discriminating test:** obtain the provider-side activity/billing record for 22:58:25Z. Zero FunctionCalls/cost supports a `$0` receipt import; any call ID/cost must be imported instead.
- **Fix/action:** append an explicit receipt/audit settlement through the existing `--import-receipt` path; never delete ledger rows.
- **Interpretability:** accounting is partial until reconciled; scientific output remains absent.

### H3 [harness | Likely | 60–70%]

- **Mechanism:** pre-dispatch client/config failures occur after reservation, so obvious local failures become unresolved spending entries and halt iteration.
- **Evidence:** reservation is created at `production.py:34`, while the explicit-run/budget check and backend callback run at lines 38–43.
- **Contrary evidence:** reserving before any potentially remote operation is the safer default, and the current halt prevented duplicate spending.
- **Discriminating test:** add a no-spend hydration/binding check before `production_stage`, while keeping reservation immediately before the actual `.remote()` call. It should fail without adding ledger rows when the app handle is unusable.
- **Fix/action:** add a pre-dispatch hydrated-handle test to the real adapter; retain reserve-before-remote semantics.
- **Interpretability:** yes for the failure, not for benchmark results.

### H4 [harness | Chances a little better than even | 45–55%]

- **Mechanism:** after Modal binding is fixed, the first strict OpenRouter JSON-schema request or remote model/container setup may fail because neither real path has yet been exercised end to end.
- **Evidence:** this attempt produced zero judge calls and zero remote model output; all prior full orchestration evidence was fake/injected.
- **Contrary evidence:** the payload schemas and fake adapters have extensive local tests, and the prior phase-6 Modal smoke loaded Qwen3.5-4B successfully.
- **Discriminating test:** after fixing binding, run one bare-only Modal condition, then one inexpensive strict-schema judge request, before the eight-method command. Each should persist a valid artifact/receipt.
- **Fix/action:** sequence the two existing production paths at smallest scale; do not change model, method, prompts, judge, or schema preemptively.
- **Interpretability:** no current scientific interpretation; these are next-risk hypotheses.

## Decision

1. **Resolve-condition verdict: not met.** The condition was “complete the real sweep, rerun from cache with no remote work, then render the report.” The first production stage failed at hydration (`...real-sweep-first-run.log:96`); no run summary exists.
2. **Prediction check:**
   - Identity/budget match before remote work: **supported** (`...dry-preflight.log:35`).
   - First live operation is bare generation: **supported** (`...real-sweep-first-run.log:36-37`).
   - A usable Modal binding creates a remote stage: **contradicted** by the hydration exception.
   - Unknown remote outcome stops rather than retries: **supported** by `costs.jsonl:1-2`.
   - Scientific method/dose predictions: **unresolved** because no model stage ran.
3. **Earliest unsupported link:** plain-process production callback → hydrated Modal function. Evidence required: one provider FunctionCall and 20 persisted bare answers from the exact callback.
4. **Validity:** “invalid” here means treating this attempt as evidence about steering methods, persona behavior, calibration transfer, or their relative quality. `P(scientific result is invalid) ≈ 0.99–1.00`; classification: **inconclusive infrastructure failure**. The failure diagnosis itself is credible.
5. **Three highest-information clues:** (1) exact `_validate_is_hydrated` traceback localizes the failure before model work; (2) empty provider app list supports zero dispatch/spend; (3) zero production-cache files/run summary rules out a hidden partial benchmark result.
6. **Missing metrics, ranked:** provider billing/FunctionCall record for the exact window; one successful bare-stage response/health artifact; strict judge-schema response and cost; candidate/transfer histories; real cache-rerun counts.
7. **Bugs requiring code changes:** H1 → explicit Modal lifecycle/hydrated handle; H3 → no-spend binding check before reservation where possible, without weakening reserve-before-remote.
8. **Misconceptions requiring reinterpretation:** passing fake callback tests did not establish that the real Modal `Function` object was hydrated from the standalone sweep process. It established payload/orchestration contracts only.
9. **What would change the verdict:** a provider record showing a remote FunctionCall would lower confidence that spend was zero; a successful bare-only run after binding would close H1; later answer/health/judge artifacts are required before any method conclusion.
10. **Recommended sequence:** reconcile reservation `f6291f...` using provider evidence; fix and test one explicit Modal app lifecycle; run only `--method bare`; then one strict judge request; only after both produce durable records resume the unchanged eight-method sweep. Combining model/method/judge changes now would destroy attribution.
