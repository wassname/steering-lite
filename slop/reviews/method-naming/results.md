# Method names

User-approved rename; PI/OpenAI.

| Previous | Current |
|---|---|
| `vjp_delta` | `vjp_resid` |
| `vjp_cache` | `vjp_value` |
| `kv_cache_gram` | `value_gram` |
| `super_sspace` | `sspace_pool` |
| `sspace_damp_amp` | `sspace_scale` |
| `svdkv` | `sink_split` |
| `svdkv_resid` | `sink_split_resid` |

Modules, config classes, method registration, maintained imports, CLI examples and current README labels use the new names. No aliases. Historical journal entries and review evidence retain the old names; `vjp_resid.py` retains the reference estimator's name, `vjp_delta`.

Verification:
- `outputs/logs/method-renames-check.log`: `55 passed in 711.44s`, followed by `SMOKE_PASS method=mean_diff rungs=2`. `just check` exited 0.
- `outputs/logs/method-renames-migration.log`: `Migrated 1100 files; verified tensor payloads and answer bytes`. Original artifacts remain in `.local/method-naming/original-artifacts/`; per-file before/after hashes are in `local-migration.json`.
- `outputs/logs/method-renames-results.log`: each of dev, full, 27b-full and olmo-full reports `scores, CIs, curves, answers and ratings unchanged`.

The result relabelling reuses saved statistics. `results.summary` shares a seeded bootstrap RNG across alphabetically sorted methods, so recomputing after a rename can alter confidence intervals through different random draws. No new judging or bootstrap was done for this rename.

- `outputs/logs/method-renames-pages.log`: four `UAT_PASS` results (dev, full, 27b-full, olmo-full).
- `verification.log`: `All 7 new CLI IDs registered, old IDs absent; 29 renamed vector configs deserialize`. All four reports' summaries, curves, method selection, colors and blind ratings exactly match pre-rename snapshots.
- `plot-review.md`: independent PNG review reports consistent naming and no clipping, with annotation/marker crowding still present. Parent inspected all four PNGs and moved the symbol key to unused lower-left space after it overlapped the dev VJP label.

Remaining: finish the Modal cache migration. The first Volume attempt verified all 1099 remote source hashes, then failed during backup copies with `ResourceExhaustedError: too many layers in volume`. Active paths were unchanged. The revised migration uses one batched upload and verifies new hashes before deleting old paths.
