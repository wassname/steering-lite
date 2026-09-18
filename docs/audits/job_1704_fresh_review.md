# Independent scientific audit: pueue job 1704

## Verdict

**This run establishes a substantial behavioral effect, not selective steering of the intended Care↑/Authority↓ contrast.** At the calibrated positive coefficient, the dominant result is increased selection of the “not morally wrong” category, with markedly worse foundation classification. The negative coefficient produces a Care-relative shift but essentially no Authority decrease.

The evidence favors **category bias and altered interpretation over coherent recovery of the intended contrast**. It does not establish that all KV-cache steering is ineffective, or that an implementation bug caused this result.

## Evidence and scope

Read completely: both supplied logs; `kv_cache_gram.py`; sweep, calibration, results, evaluation-adapter, foundation-aggregation and persona-construction implementations; predictions; repository `AGENTS.md`; ml-debug skill. The inherited sibling `AGENTS.md` does not exist.

Inspected metadata, summary statistics, initial paired evaluation traces, and all eight final calibration traces. **Did not exhaustively read every raw-score array, all 396 evaluation traces, or every intermediate calibration JSONL record.** No shell commands, numerical recomputation, tests, git diff, or staging inspection were available.

Paths below are relative to `/workspace/2026/lite/steering-lite-kv-cache`, except logs.

## 1. What the run establishes

Run identity is `c81d49454ef6`, actual commit `ef26f16` (`/tmp/job1704-full.log:1`). This differs from the prediction document’s queued commit `efd74bb`; no intervening diff was supplied.

Configuration: Qwen3-4B, rank 16, layers 7–27, one persona pair, 132 evaluation rows, 256-token reasoning budget. Despite requested/persisted `n_pairs=256`, the log states:

> “Persona-branching pairs: n=200 from 1 persona pairs”
> `/tmp/job1704-full.log:8`

Calibration genuinely reached its specified positive-arm statistic:

- `calibrated_C = 2.1579231561799768`
- `kl_at_calib = 0.7956114411354065`
- `kl_p95_at_calib = 1.456073522567749`

Source: `outputs/kv_cache_gram_r16_qwen3_4b/kv_cache_gram.json:81–85`.

The signed evaluation results are:

| Measurement | Bare | +C | −C |
|---|---:|---:|---:|
| ΔCare CLR | reference | −1.55242 | +1.71320 |
| ΔAuthority CLR | reference | −1.45062 | −0.01786 |
| ΔSocial CLR | reference | +7.80032 | +0.28186 |
| Care−Authority shift | 0 | −0.10180 | +1.73106 |
| Top-1 accuracy | 0.77273 | 0.24242 | 0.66667 |
| Informedness | 0.71183 | 0.13445 | 0.58433 |
| Mean allowed-token mass | 1.000000007 | 0.999998483 | 1.0 |
| Mean margin | 12.11884 | 11.12595 | 8.45170 |
| `wrongness = mean(1−p[social])` | 0.74271 | 0.13916 | 0.75010 |

Sources: method JSON `:87–142`, `:2256–2260`, `:2262–2316`, `:4432–4436`; bare JSON `:2226–2230`.

Thus, the positive arm loses **53.03 percentage points** of accuracy relative to bare; the negative arm loses **10.61 points**. Accuracy loss alone cannot falsify intentional value steering, but its combination with the category distribution and traces is consequential.

The method is not too weak: it reaches the KL target, changes outputs, and becomes plainly destructive at larger doses. The bracket shows RMS KL `0.3862 → 0.7956 → 1.8442 → 8.7086` at coefficients `1.5625 → 2.1579 → 3.125 → 6.25`. At 6.25 the displayed tail includes:

> “There right is the sweet sweet left. 1. 1. 1. 2. 3…”

That is evidence against the prediction’s “too weak” branch, not evidence for usable selectivity.

## 2. Concrete findings

### High — CLR category attribution is being described as moral wrongness

**Paths:** `src/steering_lite/eval/tinymfv.py:61`; `src/steering_lite/eval/foundations.py`, `dclr_per_foundation`; `scripts/results.py:231`.

Observed implementation:

> `cl = clr(np.asarray(r["score"], dtype=float))`

Every foundation is scored and aggregated across **all** rows. This is relative evidence for the forced-choice category, not an independent wrongness rating on vignettes belonging to that foundation.

Nevertheless, results prints:

> “Δclr > 0 = wrongness went UP.”

That interpretation is unsupported. Because CLR subtracts the category mean, increasing one category necessarily depresses the others. Here, the positive arm’s `+7.80032` Social CLR shift can explain substantial apparent decreases elsewhere without a selective change in attitudes to authority.

Moreover, “Social Norms” is the adapter’s name for `social`; the inspected traces identify that answer as:

> “Not morally wrong, doesn't apply to any choices.”

**Affected claim:** Care↑/Authority↓ as moral priorities, rather than a forced-choice response profile.

**Disproof check:** demonstrate the corresponding directional effects on independent wrongness/action-preference judgments, particularly matched situations where authority and welfare genuinely conflict. A category-classification change alone cannot do this.

### High — Fluent and valid answers are not evidence of semantic coherence

**Paths:** method JSON `:2256–2260`; `kv_cache_gram.pos.traces.jsonl:1–3`; `bare.traces.jsonl:1`.

The positive arm preserves near-perfect allowed-token mass and a large margin while accuracy collapses.

The first positive trace calls chuckling at an amputee:

> “a friendly or positive action”

and later:

> “The boy is not doing anything harmful.”

It assigns `p[social]=0.9999999920700114`. Bare assigns `p[care]=0.9964058786358895` on the same row.

The third positive trace reinterprets commenting on someone’s fatness as:

> “she's not being mean or bullying, but rather making a positive comment.”

These are grammatical continuations but poor grounding in the vignette. The first three positive traces also repeatedly reinterpret “This is wrong because” as a complaint about the JSON schema.

**Inference:** the positive arm causes semantic/task interpretation degradation or broad exculpatory bias. The examples do not prove its prevalence beyond the inspected rows; aggregate Social shift supports a broader category effect.

**Disproof check:** blind paired review across a balanced trace sample, separating factual fidelity, instruction comprehension, moral preference, and answer format. High margin or allowed mass cannot substitute.

### High — The displayed negative-arm result is not calibrated at the displayed KL

**Path:** `scripts/run_tinymfv_sweep.py:509–527`.

The implementation explicitly performs:

> “−C eval (same |C|, flipped sign — no recalibration)”

and prints the positive-arm `kl_hit` beside whichever arm has the largest absolute axis shift. Consequently, the final `−2.158 / 0.80` table row does **not** establish negative-arm RMS KL of 0.80.

This matters especially for the edit:

> `projection.abs()`
> `src/steering_lite/variants/kv_cache_gram.py:110`

For a unit direction, the projected coordinate obeys \(z'=z+C|z|\). At this coefficient, positive steering maps negative projections to positive ones; negative steering maps positive projections to negative ones. This is not a small symmetric amplification/damping regime.

**Affected input:** every negative-arm comparison labeled iso-KL, and any conclusion about bidirectional symmetry.

**Disproof check:** measure both signs directly, then calibrate separately if comparing at matched drift.

### Medium — Short calibration does not establish evaluation-length stability

**Paths:** `/tmp/job1704-full.log:141–203`; `src/steering_lite/calibrate.py`, `measure_kl`.

The log’s explicit expectation is:

> “SHOULD: per_t_p95 decreasing or flat across t…”

Observed p95 KL is `0.00024` at token 0, `5.26868` at token 40, and `2.73027` at token 59. This is not uniformly front-loaded or flat. It is not monotonic growth either.

Calibration covers eight prompts × 60 generated tokens, whereas inspected evaluation traces reach 256 tokens without closing reasoning. The final eight calibration continuations are readable, but all are short partial continuations, not completed task performance.

**Inference:** evaluation-length behavior is insufficiently calibrated; the profile does not prove that KL continues growing beyond 60.

**Disproof check:** both-sign 256-token measurements on unseen prompts, including the actual evaluation framing, with semantic trace inspection.

### Medium — “Held-out calibration” is not held out from extraction content

**Paths:** `scripts/run_tinymfv_sweep.py`, `_calib_prompts`; `src/steering_lite/data/personas.py`, `make_persona_pairs`.

Calibration reloads the same suffix corpus and selects eight unique `user_msg` values. Extraction samples:

> `rng.sample(entries, min(n_pairs, len(entries)))`

With 200 entries and 256 requested pairs, extraction uses all entries. Thus calibration user messages occur in extraction examples, although persona and assistant suffix framing differ.

**Affected claim:** out-of-sample calibration/generalization, not necessarily moral-evaluation leakage.

**Disproof check:** a disjoint corpus or an explicit extraction/calibration split with recorded hashes.

### Medium — Different sign-selection rules produce different “winning” arms

**Paths:** `scripts/run_tinymfv_sweep.py:522`; `scripts/results.py:86–91`.

The inline table chooses largest absolute axis shift: negative arm. Results chooses most negative Authority shift: positive arm.

Here that means the advertised green inline row and the eventual headline method select **different arms**. For the results-selected positive-minus-negative contrast, the signed on-axis mean is approximately:

\[
[(-1.55242-1.71320)-(-1.45062+0.01786)]/2=-0.91643.
\]

This is arithmetic from saved means, **not a recomputed gated-selectivity result**. I did not execute the external metric implementation.

**Disproof check:** report both arms with a prespecified sign; select sign on a separate development split. Do not equate the inline `+1.73` with headline success.

## 3. Does the method home in on the intended contrast?

**Mechanically, the implementation is a coherent contrastive cache intervention. Scientifically, this operating point does not establish the desired behavioral contrast.**

It computes per-prompt, all-token value means and second moments, projects positive-minus-negative means into the leading Gram eigenspace, then normalizes every head’s direction. No gradient objective aligns these directions to moral decisions. High value-cache energy is not evidence of task relevance; unit normalization also discards differences in head-level contrast strength.

All-token averaging includes the differing persona words themselves. Consequently, lexical/persona-format differences and downstream contextual effects can both contribute. The run provides no retained-contrast-energy, head-reliability, shuffled-label, or cache-random-direction control to distinguish them.

Positive steering produces Care↓ alongside Authority↓ and a much larger Social increase. Negative steering produces Care↑ but Authority essentially unchanged (`−0.01786`, versus reported SEM `0.24964`). Neither arm demonstrates both intended changes convincingly.

**Best supported conclusion:** substantial, asymmetric category bias with observable semantic failures at +C; a potentially useful but unvalidated Care-relative component at −C. Reject a claim of demonstrated selective moral-axis steering, not the entire cache-intervention research direction.

## 4. Cheapest decisive follow-up

### First: reuse saved traces; no model run needed

Compute paired category-transition/confusion tables stratified by `foundation_coarse`, plus forward/reverse answer-order disagreement and reasoning-closure counts. Blind-read a balanced sample across all categories.

- **Generic category-bias prediction:** many distinct ground-truth categories move to Social at +C; −C increases Care even where care is irrelevant.
- **Contrast-specific prediction:** changes concentrate in genuine authority/welfare conflicts, retain scenario facts, and do not broadly change unrelated labels.

This is the cheapest test separating category bias from conditional specificity. It cannot by itself establish moral preference.

### Then: tiny real-pipeline discriminating experiment

Use fresh, matched authority-versus-welfare conflicts plus unrelated controls; compare bare and both fixed signs, with direct action/wrongness judgments rather than only foundation labels. Keep item wording paired and blind-review complete outputs.

Measure both signs’ drift over the actual reasoning length before making matched-KL claims. Include a same-cache random direction if testing whether extraction—not merely intervention magnitude—provides specificity.

Do not begin with a large rank sweep: it would leave the principal construct-validity ambiguity unresolved.

## ml-debug audit form

| Required item | Audit result |
|---|---|
| Log/config | Complete supplied cleaned and raw logs read; cleaned log ends at approximately line 238. Qwen3-4B, rank 16, 21 layers, actual 200 pairs, 132 rows, 60-token calibration/256-token evaluation. |
| SHOULD: coherent tail | Final short continuations readable; 6.25 tail breaks. “Highest coherent” coefficient is not established because 3.125 is readable but already changes scenario interpretation. |
| SHOULD: flat/decreasing per-token KL | Not supported: p95 `0.00024` at t=0, `5.26868` at t=40, `2.73027` at t=59. |
| SHOULD: bare Care high/Sanctity lower | Numerically observed: `+3.12` versus `−1.68`; wrongness interpretation remains a construct misconception. |
| Null scales | No-op ΔCLR and KL are zero by definition. Uniform seven-way accuracy is 1/7; constant-class informedness is zero when defined. Observed bare values provide the relevant empirical baseline. No shuffled/cache-random null run supplied. |
| Initialization | No optimization/init learning curve. Persona demos recognize intended roles but truncate before completed answers. |
| Dummy comparison | Not run. Category-prior dummy is particularly important given Social concentration. |
| Baseline/held-out comparison | Bare paired evaluation available; both arms reduce label accuracy. No independent behavioral generalization dataset supplied. |
| Learning-rate schedule | Not applicable: no optimizer/backward pass. |
| Full sample viewed | Complete first bare/+C/−C JSONL records inspected, including both generated reasoning orders and score vectors; decisive excerpts above. Exact rendered evaluation input is not persisted in these records. |
| Worst step/gradients | At C=100 RMS KL `10.3815`, repetition `0.98`; no gradients by design. These numbers do not localize a defective layer. |
| Surprises | “pmass” near 1 despite +C accuracy 0.24242: explained by format validity versus semantics. “n=200” despite requested 256: explained by corpus cap. Opposite winning signs: explained by different selection rules. |
| Missing trust evidence | Full-trace aggregate audit, both-sign long-context KL, independent calibration split, random/shuffled controls, actual-run source diff, external evaluator version and smoke outputs. |
| Competing diagnoses | Subjective, nonexclusive causal possibilities summarized below. |
| Fresh review | This is the independent fresh review; no further delegation performed or permitted. |
| Cheapest separator | Existing-trace stratified transition audit, followed by tiny direct-decision contrast test. |
| Cost | Bare 342.48 s; method 1207.02 s; stage GPU memory unreported. Reuse traces before further GPU work. |

Working diagnostic allocation, not measured probabilities:

- **50% extraction/shortcut confound:** all-token lexical/persona contrast plus broad sign-rectified intervention; supported by cross-category Social shift. Against: opposite arm retains considerable discrimination.
- **25% evaluation/task-framing failure:** category CLR is interpreted as preference; schema-confusion text is visible. Against: bare performs substantially better under the same framing.
- **10% cache implementation/path bug:** possible without cached-versus-full-forward equivalence evidence for this exact run. Against: source is mathematically consistent and calibration is dose-responsive; no observed cache exception.
- **15% unknown/mixed causes:** insufficient control interventions and held-out behavioral evidence.

Three ways a blanket negative conclusion could be false: a smaller dose may preserve specificity; a direct moral-choice task may reveal effects obscured by category classification; alternative extraction/rank/head weighting may recover the intended direction. Each requires its own controlled test. None rescues a success claim for this run as presently measured.
