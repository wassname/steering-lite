# vjp-steering reference pipeline comparison

Scope: source-only comparison. The reference is the pinned submodule `docs/vendor/vjp-steering` at `7f0782afd42565a845fcfef3d764030168fdeb2b`. Its configured remote is `https://github.com/wassname/vjp-steering.git`. No submodule code or dependencies were run. No paid work ran during this audit.

The active plan requires the reference as a base, but also specifies a 20-question, four-case RMS-KL, cached and budget-bounded comparison:

> “Use Terra or DeepSeek V4 Flash workers, not Astra” ([plan](../../.pi/plan/a565e7-v1.md#L11)).
>
> “Preserve the existing aware judge protocol and separate blind call. Score every measured dose as directed intended effect − 4 × off-target effect” ([plan](../../.pi/plan/a565e7-v1.md#L12)).
>
> “evaluate 0.8×/1.0×/1.2× predicted doses” ([plan](../../.pi/plan/a565e7-v1.md#L14)).

## Observed judge-model mismatch

The reference default judge is `deepseek/deepseek-v4-flash-0731`; the active implementation specifies `deepseek/deepseek-chat`.

> `MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-v4-flash-0731")` ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L16)).
>
> `JUDGE_MODEL = "deepseek/deepseek-chat"` ([active sweep](../../src/steering_lite/benchmark/sweep.py#L27)).

Classification: invented relative to the pinned reference; no approving plan line was found. The source also names older reference provenance, `efcd848`, while the pinned reference is `7f0782…` ([active judge](../../src/steering_lite/benchmark/judge.py#L1-L4)).

No judge-model, provider, cache or payload change was made in this audit.

## Behavior comparison

| Behavior | Reference | Active implementation | Classification |
|---|---|---|---|
| Model and cohort | Qwen3.5-4B, 100 questions, three extraction seeds ([reference walk](../../docs/vendor/vjp-steering/scripts/walk.py#L193-L205); [reference results](../../docs/vendor/vjp-steering/src/vjp_steering/results.py#L61-L65)). | Qwen3.5-4B, 20 numbered evaluation questions and one extraction path. | Intended plan change. The plan explicitly requests 20 questions and a low-cost run. |
| Methods | `vjp_delta`, mean difference, PCA and random ([reference results](../../docs/vendor/vjp-steering/src/vjp_steering/results.py#L16-L28)). | Those controls plus bare, prompting, KV-cache Gram and VJP-cache ([active pipeline](../../src/steering_lite/benchmark/pipeline.py#L12-L15)). | Intended plan change. |
| Persona pair and generation prompt | `sycophantic` / `abrasive`, `Answer as someone who is {persona}.`, thinking disabled and two short sentences ([reference walk](../../docs/vendor/vjp-steering/scripts/walk.py#L25-L34); [reference walk](../../docs/vendor/vjp-steering/scripts/walk.py#L365-L405)). | Same persona text in the sweep ([active sweep](../../src/steering_lite/benchmark/sweep.py#L33-L39)); the remote run records a two-sentence, thinking-disabled prompt specification ([active entrypoint](../../scripts/run_bsbench_sweep.py#L288-L305)). | Copied. |
| Aware judge prompt and schema | Per-response on-axis and off-axis ratings, strict `demo_rating` JSON ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L46-L55); [reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L218-L254)). | Same rubric/schema structure, with an explicit candidness sign for the selected target ([active judge](../../src/steering_lite/benchmark/judge.py#L84-L129)). | Copied with a small, visible prompt change. The sign clarification is not present in the reference. |
| Blind protocol | Reference evaluates AB/BA and two stochastic passes, but has no target-blind request ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L195-L201)). | Active code emits one aware and one target-blind strict-schema request for each AB and BA order, with method/side/coefficient omitted from the blind record ([active validation](../../src/steering_lite/benchmark/validation.py#L38-L54); [active judge](../../src/steering_lite/benchmark/judge.py#L132-L196)). | Intended plan change. The separate blind call is explicitly required at plan line 12 and by the user’s blind-concept request. |
| Judge dispatch, pacing and retry | Async concurrency six; three immediate transient attempts; may skip malformed or repeated cells ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L257-L309)). | Sequential requests, 10-second pacing and failure propagation. Every request is reserved and cached separately ([active entrypoint](../../scripts/run_bsbench_sweep.py#L106-L174); [active adapters](../../src/steering_lite/benchmark/adapters.py#L136-L169)). | Intended plan change. The active plan requires a paid-stage stop, exact reconciliation and only affected-stage retry. |
| Judge provider and model | OpenRouter base URL and DeepSeek V4 Flash default ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L16); [reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L362-L370)). | OpenRouter endpoint, but DeepSeek Chat and DeepSeek-Chat prices ([active sweep](../../src/steering_lite/benchmark/sweep.py#L26-L47)). | Provider copied; judge model and prices invented. This is the blocking mismatch. |
| Scoring and health | Per-scenario AB/BA/two-pass aggregation, signed by side; `admissible` also requires off-axis score at most 1.5 ([reference export](../../docs/vendor/vjp-steering/scripts/export.py#L147-L198)). The table selects each directional peak then computes weaker-direction effect minus damage ([reference results](../../docs/vendor/vjp-steering/src/vjp_steering/results.py#L274-L314)). | AB/BA mean, absolute off-target effect, and `directed − 4 × absolute off-target`; coherence is only generation-health reasons ([active results](../../src/steering_lite/benchmark/results.py#L55-L76)). | Intended plan change. Plan line 13 forbids an invented off-axis coherence cutoff and line 12 mandates 1:4 scoring. |
| Directional final evaluation | Reference generates bare, `+C`, and `-C` at every dose ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L160-L182)) and selects a result in each direction ([reference results](../../docs/vendor/vjp-steering/src/vjp_steering/results.py#L281-L302)). | Final rows are hard-coded with `side: "+C"` ([active production](../../src/steering_lite/benchmark/production.py#L243-L265)); result grouping reconstructs only `+C` comparisons ([active results](../../src/steering_lite/benchmark/results.py#L165-L180)). | Missing relative to the reference. No plan line approving removal of the negative-direction final arm was found. |
| Calibration and transfer | Reference sweeps a 100-question dose grid and detects health breakdown ([reference walk](../../docs/vendor/vjp-steering/scripts/walk.py#L193-L285)). | Four fixed calibration questions, maximum healthy candidate, RMS-KL target, four disjoint cases, and 0.8/1.0/1.2 predicted doses ([active dose search](../../src/steering_lite/benchmark/dose_search.py#L18-L37); [active dose search](../../src/steering_lite/benchmark/dose_search.py#L68-L79); [active dose search](../../src/steering_lite/benchmark/dose_search.py#L145-L169); [active dose search](../../src/steering_lite/benchmark/dose_search.py#L200-L205)). | Intended plan change. |
| Cache identity | Reference hashes bare/steered text, prompt, target, rubric, model, order and pass ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L195-L215)). | Local stage cache hashes model, data, method, config, prompts and source; judge cache additionally includes the complete request, endpoint, model and cost upper ([active cache](../../src/steering_lite/benchmark/cache.py#L43-L87); [active adapters](../../src/steering_lite/benchmark/adapters.py#L136-L166)). | Intended auditability change. It is not copied byte-for-byte, but it preserves the reference’s content-keyed principle and adds plan-required cost/provenance inputs. |
| Spending and provider provenance | Reference records provider and per-response reported cost in a JSONL cache ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L334-L353)). | Local code requires budget preflight, reservations, receipts/settlement, provider timing evidence and response usage ([active sweep](../../src/steering_lite/benchmark/sweep.py#L157-L179); [active adapters](../../src/steering_lite/benchmark/adapters.py#L102-L115); [active entrypoint](../../scripts/run_bsbench_sweep.py#L106-L174)). | Intended plan change for the $50 limit and failure reconciliation. |
| Results and figures | Reference reads a CSV, averages seeds, creates one Plotly PNG plus HTML/Markdown table parity ([reference results](../../docs/vendor/vjp-steering/src/vjp_steering/results.py#L41-L83); [reference results](../../docs/vendor/vjp_steering/results.py#L429-L461)). | Active renderer admits only a complete eight-method summary; it makes `measured-points.json`, two PNGs, HTML tables and numbered aware/blind/health evidence ([active results](../../src/steering_lite/benchmark/results.py#L230-L274); [active results](../../src/steering_lite/benchmark/results.py#L337-L388); [active results](../../src/steering_lite/benchmark/results.py#L391-L440)). | Intended plan change. Current final report remains unavailable because VJP-delta and VJP-cache are incomplete, not because the renderer uses calibration points. |

## Mechanical consequences of the observed identities

- Reference cache identity includes `model`, `order`, `pass`, rubric, target and hashes of bare/steered/prompt text ([reference judge](../../docs/vendor/vjp-steering/scripts/judge.py#L91-L104)).
- Active judge cache identity includes the complete request, model, endpoint and upper cost ([active adapters](../../src/steering_lite/benchmark/adapters.py#L136-L166)). A different model or request payload therefore produces a different key. This states cache behavior only; no retention or rerun choice is made here.
- Active cost estimates use DeepSeek-Chat-specific prices and source URL ([active sweep](../../src/steering_lite/benchmark/sweep.py#L41-L47)). A changed model would require a different estimate before the current budget code could describe it. No model or estimate is selected here.
- Tests contain explicit `deepseek/deepseek-chat` expectations (for example [sweep test](../../tests/test_benchmark_sweep.py#L166)); a model-constant change would mechanically invalidate those assertions. No test edit was made.

## Status

- Scheduled VJP-delta retry `task_mubmgywg_61wedb` was cancelled before this audit.
- No provider request, Modal call, ledger event, source-policy edit, or cache-identity edit occurred here.
- Submodule status is clean at the pinned SHA above.

<!-- PI[gpt-5.6-terra] 2026-09-21 -->
