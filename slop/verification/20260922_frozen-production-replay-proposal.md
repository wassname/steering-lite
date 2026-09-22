# Frozen production replay proposal

Prepared by PI/OpenAI. Not executed. Requires parent authorization after the paid process is terminal and its ledger resolved.

This invokes the actual `run_full_sweep` with the real adapters, original model/endpoint/prompt identity and current pricing. Only outgoing callback bodies and reservation entry points are replaced in memory with counters that raise. A missed cache therefore fails before either payment or ledger mutation. No credential or Modal app is needed for this no-dispatch proof. This is the production orchestrator replay, not an end-to-end exercise of CLI authentication/app startup.

Runtime attachment points: `scripts/run_bsbench_sweep.py::run_full_sweep`; `adapters.real_adapters` callback arguments; `production.reserve` and `adapters.reserve`. No file patches.

```bash
set -o pipefail
.venv/bin/python - <<'PY' 2>&1 | tee slop/verification/20260922_frozen-production-replay.log
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from threading import Lock

from steering_lite.benchmark import adapters, production
from steering_lite.benchmark.cache import content_key, require_resolved_ledger, source_hash
from steering_lite.benchmark.generation import read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.sweep import dry_manifest, load_judge_pricing

root = Path('outputs/bsbench-v2')
ledger = root / 'costs.jsonl'
base = Path('slop/verification/20260922_frozen-production-replay')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == 'c7d23dd06fe818d4479a9539825a7fa98dd749c1'
assert not subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts'], text=True)
require_resolved_ledger(ledger)
prior = json.loads((root / 'run-summary.json').read_text())
assert prior['methods'] == list(METHODS)
assert set(prior['conditions']) == set(METHODS)
assert prior['identity_sha256'] == content_key(prior['identity'])
load_judge_pricing(Path('slop/verification/20260922_v4-provider-endpoint-metadata.json'))
identity = prior['identity']
model = identity['model']
endpoint = identity['judge']['endpoint']
budget = dry_manifest(root, model['id'], ledger=ledger, cache_aware=True, judge_endpoint=endpoint)['cost_estimate']
assert budget['paid_preflight_passed']

spec = importlib.util.spec_from_file_location('frozen_bsbench_entrypoint', 'scripts/run_bsbench_sweep.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
counts = {'gpu': 0, 'judge': 0, 'reservation': 0}
lock = Lock()
def forbidden(kind):
    def call(*args, **kwargs):
        with lock:
            counts[kind] += 1
        raise AssertionError(f'Frozen replay attempted {kind}')
    return call
production.reserve = forbidden('reservation')
adapters.reserve = forbidden('reservation')
backend, judge = adapters.real_adapters(
    modal_stage_call=forbidden('gpu'), judge_request_call=forbidden('judge'),
    judge_endpoint=endpoint, explicit_run=True, budget_preflight=budget,
    root=root, ledger=ledger,
)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot():
    return {
        'source_sha256': source_hash(),
        'ledger_sha256': digest(ledger),
        'ledger_bytes': ledger.stat().st_size,
        'provider_files': {str(p): digest(p) for p in sorted((root / 'provider-evidence').rglob('*')) if p.is_file()},
        'cache_files': {str(p): digest(p) for p in sorted((root / 'cache').rglob('*')) if p.is_file()},
        'vector_files': {str(p): digest(p) for p in sorted((root / 'artifacts' / 'vectors').rglob('*')) if p.is_file()},
        'summary_sha256': digest(root / 'run-summary.json'),
    }
def scientific(value):
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k != 'reused'}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value
before = snapshot()
Path(str(base) + '-before.json').write_text(json.dumps(before, indent=2) + '\n')
passed = False
try:
    entry.run_full_sweep(root, ledger, model=model, rows=read_dev_cohort(),
                         backend=backend, prompt_spec=identity['prompt_spec'], judge=judge)
    after = snapshot()
    for key in before:
        if key != 'summary_sha256':
            assert before[key] == after[key], key
    current = json.loads((root / 'run-summary.json').read_text())
    assert current['identity'] == identity
    assert scientific(current) == scientific(prior)
    assert counts == {'gpu': 0, 'judge': 0, 'reservation': 0}
    require_resolved_ledger(ledger)
    passed = True
finally:
    after = snapshot()
    Path(str(base) + '-after.json').write_text(json.dumps(after, indent=2) + '\n')
    proof = {'author': 'PI/OpenAI', 'passed': passed, 'callback_attempts': counts,
             'identity_sha256': prior['identity_sha256'],
             'ledger_unchanged': before['ledger_sha256'] == after['ledger_sha256'],
             'summary_comparison_ignores_only_runtime_reused_flags': True}
    Path(str(base) + '-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))
PY
```

`reused` is a runtime flag added by `production_stage`, so the summary may legitimately change false→true after replay. Scientific data and identity must remain equal after removing that flag; stage caches, vectors, provider evidence and ledger must remain byte-identical. Guards stop a missing paid stage before reservation, so zero counters plus successful complete traversal prove that the real adapters needed no dispatch.

## Rendering without changing scientific cache identity

`cache.source_hash()` hashes **every `*.py` under `src/steering_lite`**, including `benchmark/results.py`, into all stage identities. `run_full_sweep` additionally hashes `scripts/run_bsbench_sweep.py` into the summary identity. Consequently editing the existing package renderer, excluding it from `source_hash`, or editing the sweep entrypoint changes the source identity. Do none of these during replay.

After successful replay, a reporting-only implementation in the existing `scripts/run_bsbench_results.py` (and an external-to-package template if needed) can read the frozen summary and cache artifacts without changing either hash. That script is not in either hash's input set. It must retain the summary's scientific identity rather than relabeling it with its own reporting revision; record a separate renderer hash for provenance. Its current import of the stale package renderer would need replacement, but no execution/scoring source or identity override is required. Do not invoke `run_full_sweep` from rendering. Parent decides this source window; no renderer change has been made.
