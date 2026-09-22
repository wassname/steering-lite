"""Budgeted one-off mean_diff extraction-cap diagnostic; no dispatch without --run and exact ID. — PI/gpt-6-sol"""
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
OLD_SHA256 = '169c8ed960bb14f24ec8deb59714c7eeb34e7abae63f804f982383eb9aaf5d8d'
OUTPUT = ROOT / 'slop/verification/20260923_mean_diff_cap_diagnostic'
LEDGER = ROOT / 'outputs/bsbench-v2/costs.jsonl'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare() -> tuple[dict, bytes, dict, dict]:
    from transformers import AutoTokenizer
    from steering_lite.benchmark.cache import committed, content_key, require_resolved_ledger, source_hash
    from steering_lite.benchmark.dose_search import BENCHMARK_KL_SPEC
    from steering_lite.benchmark.generation import generation_inputs, read_dev_cohort
    from steering_lite.benchmark.sweep import (
        BUDGET_LIMIT_USD, MODAL_GPU_STAGE_UPPER_USD, PHASE6_SMOKE_LEDGER,
        persona_extraction_identity, phase6_smoke_committed,
    )

    require_resolved_ledger(PHASE6_SMOKE_LEDGER)
    snapshot = Path('/home/code/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots') / REVISION
    if not snapshot.is_dir():
        raise ValueError('the local pinned Qwen snapshot is missing')
    model_config = json.loads((snapshot / 'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if tokenizer.truncation_side != 'right':
        raise ValueError('the local pinned tokenizer truncation side changed')
    summary = json.loads((ROOT / 'outputs/bsbench-v2/run-summary.json').read_text())
    candidate = summary['conditions']['mean_diff']['candidate']
    artifact = candidate['vector_artifact']
    if artifact['sha256'] != OLD_SHA256 or candidate['vector_sha256'] != OLD_SHA256:
        raise ValueError('historical mean_diff vector SHA differs from the pinned diagnostic')
    old_bytes = (ROOT / 'outputs/bsbench-v2' / artifact['path']).read_bytes()
    if len(old_bytes) != artifact['bytes'] or _sha(old_bytes) != OLD_SHA256:
        raise ValueError('historical mean_diff vector bytes differ from the sidecar')
    rows = read_dev_cohort()[:4]
    prompts = [row['prompt'] for row in rows]
    formatted = generation_inputs(tokenizer, rows)
    if summary['identity']['model']['id'] != 'Qwen/Qwen3.5-4B':
        raise ValueError('historical model ID differs')
    prompt_spec = summary['identity']['prompt_spec']
    if prompt_spec != {'template': 'Answer in 2 short sentences.', 'enable_thinking': False, 'max_new_tokens': 128}:
        raise ValueError('historical generation settings differ')
    historical = {'bare': candidate['baseline_answers'], 'answers_by_side': {}}
    if len(historical['bare']) != 4:
        raise ValueError('historical baseline answer count differs')
    for side in ('+C', '-C'):
        items = sorted((item for item in candidate['candidate_items'] if item['side'] == side and item['magnitude'] == 0.8), key=lambda item: item['prompt_index'])
        if [item['prompt_index'] for item in items] != list(range(4)):
            raise ValueError(f'historical {side} 0.8 candidate is incomplete')
        if [item['prompt_sha256'] for item in items] != [_sha(prompt.encode()) for prompt in prompts]:
            raise ValueError(f'historical {side} calibration prompts differ')
        historical['answers_by_side'][side] = [item['response'] for item in items]
    persona = persona_extraction_identity()
    if candidate['method_config']['method'] != 'mean_diff' or candidate['method_config']['layers'] != [7, 11, 15, 19, 23]:
        raise ValueError('historical mean_diff method config differs')
    spec = {
        'schema': 'bsbench-mean-diff-cap-diagnostic-v1', 'method': 'mean_diff',
        'source_sha256': source_hash(),
        'callback_sha256': _sha((ROOT / 'slop/verification/20260923_mean_diff_cap_modal.py').read_bytes()),
        'model_id': 'Qwen/Qwen3.5-4B', 'model_revision': REVISION,
        'snapshot_sha256': {file: _sha((snapshot / file).read_bytes()) for file in ('config.json', 'tokenizer_config.json')},
        'layer_types': model_config['text_config']['layer_types'], 'target_layer': 29,
        'old_vector_sha256': OLD_SHA256, 'old_vector_config': candidate['method_config'],
        'persona_identity': persona, 'prompts': prompts,
        'formatted_prompt_sha256s': [_sha(prompt.encode()) for prompt in formatted],
        'historical_answers_sha256': content_key(historical),
        'coefficients': {'+C': 0.8, '-C': -0.8},
        'max_new_tokens': 128, 'kl_spec': BENCHMARK_KL_SPEC,
        'new_extraction_max_length': 384,
    }
    spec['diagnostic_id'] = content_key(spec)
    ledger_usd = committed(LEDGER)
    external_usd = phase6_smoke_committed()
    budget = {
        'existing_ledger_usd': ledger_usd, 'external_committed_usd': external_usd,
        'existing_committed_usd': ledger_usd + external_usd,
        'stage_upper_usd': MODAL_GPU_STAGE_UPPER_USD,
        'total_upper_usd': ledger_usd + external_usd + MODAL_GPU_STAGE_UPPER_USD,
        'limit_usd': BUDGET_LIMIT_USD, 'estimated_judge_usd': 0.0,
        'planned_gpu_jobs': 1,
        'paid_preflight_passed': ledger_usd + external_usd + MODAL_GPU_STAGE_UPPER_USD < BUDGET_LIMIT_USD,
    }
    return spec, old_bytes, historical, budget


def _record_receipt(ledger: Path, reservation: str, raw: dict) -> None:
    from steering_lite.benchmark.cache import estimate_at_reservation_upper, mark_unresolved

    receipt = raw['cost_receipt']
    if receipt['provider'] != 'Modal' or receipt['status'] != 'pending' or receipt['usage']['elapsed_seconds'] <= 0:
        mark_unresolved(ledger, reservation, 'diagnostic_receipt_invalid')
        raise ValueError('diagnostic GPU returned an invalid usage receipt')
    estimate_at_reservation_upper(ledger, reservation, receipt)


def _review(raw: dict, spec: dict, historical: dict) -> dict:
    if raw['diagnostic_id'] != spec['diagnostic_id'] or raw['source_sha256'] != spec['source_sha256']:
        raise ValueError('diagnostic output identity differs')
    if raw['model_revision'] != spec['model_revision'] or raw['snapshot_sha256'] != spec['snapshot_sha256']:
        raise ValueError('diagnostic output model/tokenizer attestation differs')
    if raw['old_vector_sha256'] != spec['old_vector_sha256']:
        raise ValueError('diagnostic output old vector differs')
    if len(raw['bare_before']) != 4 or any(len(raw['measurements_by_cap'][label][side]['answers']) != 4 for label in ('old64', 'new384') for side in ('+C', '-C')):
        raise ValueError('diagnostic must retain all four matched prompts for both sides')
    if _sha(base64.b64decode(raw['new_vector_bytes_b64'])) != raw['new_vector_sha256']:
        raise ValueError('new extraction vector bytes differ')
    if raw['bare_before'] != raw['bare_after']:
        raise ValueError('steering attachment leaked into the bare after-check')
    return {
        'schema': 'bsbench-mean-diff-cap-diagnostic-review-v1',
        'diagnostic_id': spec['diagnostic_id'],
        'bare_restored': True,
        'historical_drift': {
            'bare': [index for index, (before, old) in enumerate(zip(raw['bare_before'], historical['bare'], strict=True)) if before != old],
            **{side: [index for index, (fresh, old) in enumerate(zip(raw['measurements_by_cap']['old64'][side]['answers'], historical['answers_by_side'][side], strict=True)) if fresh != old]
               for side in ('+C', '-C')},
        },
        'direction_comparison': raw['direction_comparison'],
        'raw_path': str(OUTPUT / spec['diagnostic_id'] / 'raw.json'),
    }


def run(spec: dict, old_bytes: bytes, historical: dict, budget: dict, approved_id: str) -> None:
    if approved_id != spec['diagnostic_id']:
        raise ValueError('paid dispatch requires the exact inspected diagnostic ID')
    from steering_lite.benchmark.cache import mark_unresolved, require_resolved_ledger, reserve_many
    from steering_lite.benchmark.sweep import BUDGET_LIMIT_USD, phase6_smoke_committed, preflight_budget

    out = OUTPUT / spec['diagnostic_id']
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw_path = out / 'raw.json'
        kind = f"modal-cap-diagnostic-mean-diff-{spec['diagnostic_id']}"
        records = [json.loads(line) for line in LEDGER.read_text().splitlines()]
        reservations = [row for row in records if row['event'] == 'reserved' and row['kind'] == kind]
        if len(reservations) > 1:
            raise RuntimeError('duplicate diagnostic reservations require an audit')
        reservation = reservations[0]['id'] if reservations else None
        if not raw_path.exists():
            import importlib.util

            callback_path = ROOT / 'slop/verification/20260923_mean_diff_cap_modal.py'
            if _sha(callback_path.read_bytes()) != spec['callback_sha256']:
                raise ValueError('diagnostic callback changed after its output identity was built')
            definition = importlib.util.spec_from_file_location('bsbench_cap_diagnostic_modal', callback_path)
            callback = importlib.util.module_from_spec(definition)
            definition.loader.exec_module(callback)
            try:
                recovered = b''.join(callback.diagnostics.read_file(f"{spec['diagnostic_id']}.json"))
            except FileNotFoundError:
                if reservation is not None:
                    raise RuntimeError('diagnostic already reserved; inspect the remote job before another dispatch')
                require_resolved_ledger(LEDGER)
                preflight_budget(
                    LEDGER, {'total_upper_usd': budget['stage_upper_usd']},
                    external_committed_usd=phase6_smoke_committed(),
                )
                reservation, = reserve_many(
                    LEDGER, [(kind, budget['stage_upper_usd'])],
                    limit_usd=BUDGET_LIMIT_USD - phase6_smoke_committed(), strict_limit=True,
                )
                try:
                    with callback.app.run():
                        raw = callback.compare_mean_diff_caps.remote(spec, base64.b64encode(old_bytes).decode())
                        _save(raw_path, raw)
                except Exception:
                    mark_unresolved(LEDGER, reservation, 'diagnostic_dispatch_or_local_persist_failure')
                    raise
            else:
                temporary = raw_path.with_suffix('.tmp')
                temporary.write_bytes(recovered)
                temporary.replace(raw_path)
        raw = json.loads(raw_path.read_text())
        if reservation is None:
            raise RuntimeError('diagnostic output lacks its cost reservation')
        _record_receipt(LEDGER, reservation, raw)
        review = _review(raw, spec, historical)
        _save(out / 'review.json', review)
        print(json.dumps({'raw_path': str(raw_path), 'review': review}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--approved-id', help='Exact diagnostic ID inspected and authorized by the parent')
    args = parser.parse_args()
    spec, old_bytes, historical, budget = prepare()
    if args.dry_run:
        if args.approved_id is not None:
            raise ValueError('--dry-run cannot authorize dispatch')
        print(json.dumps({'spec': spec, 'budget': budget, 'paid_dispatch': False}, indent=2))
        return
    if args.approved_id is None:
        raise ValueError('--run requires --approved-id')
    run(spec, old_bytes, historical, budget, args.approved_id)


if __name__ == '__main__':
    main()
