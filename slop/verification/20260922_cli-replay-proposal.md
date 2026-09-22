# No-dispatch CLI / just sweep verification proposal

Prepared by PI/OpenAI; not executed. The accepted replay covered the actual orchestrator and real adapters, but did not exercise CLI parsing, authentication checks or `Modal app.run()` startup. `just results` is a different, offline recipe and cannot establish those checks.

## Exact entry points

Both must use the existing frozen source, endpoint pricing artifact, output directory and real adapter types:

```bash
.venv/bin/python scripts/run_bsbench_sweep.py --run --backend real \
  --judge-pricing slop/verification/20260922_v4-provider-endpoint-metadata.json

just sweep '--run --backend real --judge-pricing slop/verification/20260922_v4-provider-endpoint-metadata.json' \
  'Qwen/Qwen3.5-4B' 'outputs/bsbench-v2' '.local/bsbench-cli-proof/no-shell-env'
```

These commands are proposals, not permission to run them without guards. The final `just` argument names a nonexistent file so the recipe does not shell-source `.env`. The Python bootstrap below loads the existing project credential through python-dotenv; it never prints or copies it. Do not create the `no-shell-env` path.

## Runtime guard preparation

Parent can authorize a temporary `.local/bsbench-cli-proof/sitecustomize.py`, outside scientific hash inputs, and prepend its absolute directory plus the existing `src` directory to `PYTHONPATH`. Before either command runs, this bootstrap must:

1. Load dotenv through `find_dotenv(usecwd=True)` / `load_dotenv`, asserting the key is nonempty without printing it.
2. Import actual `steering_lite.benchmark.adapters`, `production`, and `run_bsbench_modal` after inserting the absolute `scripts` directory in `sys.path`.
3. Replace `adapters.openrouter_request_callback` with a factory returning a counting/raising callback. Replace `run_bsbench_modal.remote_stage_call` the same way. Keep their caller signatures (`**kwargs`) and actual real adapters unchanged.
4. Replace `production.reserve` and `adapters.reserve` with a counting/raising guard. The first missing cache must fail before any ledger write.
5. Register an `atexit` writer for a separate per-command counts file. Restrict bootstrap activation to the sweep entrypoint process, so any unrelated subprocess cannot overwrite counts.
6. Enclose bootstrap setup in a fail-closed exception handler using `os._exit(99)` after a bounded setup-error message. Python otherwise prints and ignores errors in `sitecustomize`, which would be unsafe here.

Leave `run_bsbench_modal.app.run()` unchanged. Thus any actual Modal startup error is observed, not hidden by a dummy app. This can contact Modal's control plane but must dispatch no GPU function and make no OpenRouter completion request. If no control-plane access is authorized, retain the narrower accepted orchestrator proof and state that CLI startup remains unverified.

Before each command, capture source/entrypoint/summary identity and ledger/provider/cache/vector hashes as in the accepted replay. Save command exit code, complete stdout/stderr, zero attempted callbacks/reservations and after hashes. Compare summary excluding only runtime `reused`. A process exit0 without the bootstrap counts file is failure, not success. No callback guard may merely log and continue. Afterward remove the bootstrap from `PYTHONPATH`; preserve its text and proof in `slop/verification` if used.

No bootstrap file or environment change has been applied by this proposal. The parent owns authorization and startup judgment.
