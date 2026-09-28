# Round 1 synthesis (PI/Claude, 2026-09-28)

Reviews: round1-code-kimi.md (BLOCK), round1-claims-kimi.md (OK with notes), round1-oracle-gpt.md (second opinion).

## P1, fix now

1. Legacy vectors lack "thinking", so the cache check in walk.py extract_vector defaults it to the current run's value: `--no-think` without `--tag` still loads a thinking=True vector (code review, repro given). Fix: backfill "thinking" into every existing vector json (local and Modal Volume `steering-lite-bsbench-v3`, path `bsbench/<model>-g7c7712c6/vectors/*.json`): true for all, false for `vjp_delta-nothink_s0.json` (the only no-think extraction). Then read it as a required key (no default).
2. Diagnostic mode outputs (profile/, vjp_check/, vjp_split/) are keyed only on method name or not at all (profile), so a changed setting reuses an old file (run_modal.py cached_on_volume) or overwrites it (walk.py). Fix: one rule in walk.parse_args: if any extraction setting differs from its parser default (layers, target_layer, n_pairs, max_length, no_think) and no --tag is given, parser.error. And one function in walk.py returning each mode's output path, including the tag for profile (`profile/persona-<tag>_s<seed>.json` when tagged, unchanged `persona_s<seed>.json` otherwise), used by both walk.py and run_modal.py.

## P2, small, do now

3. results.py spare colour `pop(0)` on an empty list: fail with a clear message instead of IndexError.
4. Journal citations (claims review): 4B "-1.16 at KL 0.65" cites olmo-full/points.json, source is results/full/points.json; "vector cos min 0.979 median 0.995" has no source file: save the computation output to a file and cite it.
5. olmo_read_25q.md: note "#16 for vjp_cache worse than bare" contradicts its row (both A): reword as "more invented detail", not a stance change.
6. By-stance journal entry: add the caveats from the oracle and claims review: n counts question x seed (27B accepts = 21 unique questions, correlated over 3 seeds), no CI; the dose was chosen on the same questions; the bare-accept subsets differ across models, so this is a within-model headroom check, not a controlled cross-model comparison; selecting on a noisy bare rating risks regression to the mean. Also cite the +C side support (27b-full/index.md).

## Report only / deferred

- Wasted bare generation before --vjp-split/--vjp-check: bare is cached and shared, cost is one pass per model. Deferred.
- Odd n_pairs drops one pair in --vjp-split: n is 200. Ignore.
- Oracle: the VJP estimator is a difference of gradients over interior positions, not a behavioural gradient; suggests a finite-difference audit. This is a next experiment, not a fix to this diff; offered to the user.
- Oracle: the hand reading used vjp_delta-nothink, not default vjp_delta: already stated in olmo_read_25q.md; the vectors have cos 0.995.

## Round 2 (targeted follow-up on the fix diff round2-fix.diff)

round2-code-kimi.md: both P1 resolved, no invocation used so far is rejected, untagged cache names kept; "Fix verdict: RESOLVED", "Merge verdict: OK with notes". round2-claims-gpt.md: items 4-6 resolved, numbers match sources; "Fix verdict: RESOLVED", "Merge verdict: OK with notes". Loop stopped: no P0, no P1. Remaining notes: the extra bare generation before --vjp-split/--vjp-check (cached after the first run); the oracle's finite-difference audit (a next experiment); the interpretation in the journal stays inference.
