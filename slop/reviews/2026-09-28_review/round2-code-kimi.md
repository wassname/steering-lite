All checks complete. Findings:

**(1) Both P1s resolved.**

- **P1 no-think/cache reuse — RESOLVED, two layers.** (a) `parse_args` now rejects any changed extraction setting (`--no-think`, `--layers`, `--target-layer`, `--n-pairs`, `--max-length`) without `--tag`/`--smoke` (walk.py:103-107). Repro: `walk.parse_args(['vjp_delta','--no-think'])` → `SystemExit 2` with "add --tag". (b) `extract_vector` reads `saved["thinking"]` as a required key (walk.py:155) and the backfill was actually applied: I verified all vector sidecars — `vjp_delta_s0.json` has `thinking: true`, `vjp_delta-nothink_s0.json` has `false`, t47/t48 `true` (both models). So the vacuous-default hole is gone for the vectors that exist; a settings mismatch now raises.
- **P1 mode-cache keys — RESOLVED.** run_modal.py:70 `cached_on_volume` now reads `walk.mode_output(args)` (run_modal.py:70), the same function walk.py writes through (walk.py:483, 512, 517), so reader/writer keys cannot drift; tagged profile runs get `persona-<tag>_s<seed>.json`, and `--profile --no-think` without a tag is rejected at parse time.

**(2) No new defect in the blast radius.**

- I replayed every recorded real argv from the DONE lines in `outputs/logs/modal-*.log` and `vjp-split-*.log`: the vjp-split runs passed only `--vjp-split --extract-batch-size 2` (n_pairs=200 was then the default — not a CLI flag, so not "changed"); nothink/t47/t48 all carry `--tag`; the exact t47 argv parses today (repro above, name `vjp_delta-t47`). justfile `smoke-bsbench` and run_modal `smoke()` carry `--smoke`. No existing invocation is rejected.
- `--smoke` exemption is sound: smoke redirects OUT to the throw-away `outputs/bsbench-smoke` tree (walk.py:588), and the strict sidecar check in `extract_vector` still runs there, so a mismatched smoke vector fails rather than silently reuses.
- `mode_output` keeps untagged names: `profile/persona_s{seed}.json`, `vjp_split|vjp_check/{method}_s{seed}.json` (verified by repro); matches existing `vjp_split/vjp_delta_s0.json` files, so pre-fix cached diagnostics are still recognized by `cached_on_volume`.
- Note (not a blocker): `parser.error` raises SystemExit inside `cached_on_volume` (run_modal.py:63), so a changed-settings/no-tag argv now kills the Modal local entrypoint with exit 2 instead of returning a cache verdict. Clear message, fail-fast per AGENTS.md.

**(3) P2 notes.** `spare.pop(0)` fixed — results.py:558-560 raises a ValueError naming the uncoloured methods. Wasted bare generation before the `vjp_split`/`vjp_check` branches still stands (bare at walk.py:490, branches at 509/516) — cosmetic, cached after first run.

Fix verdict: RESOLVED
Merge verdict: OK with notes