"""Fail closed if the diagnostic preflight attempts a network connection. — PI/gpt-6-sol"""
import contextlib
import hashlib
import io
import json
import os
import runpy
import socket
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
script = root / 'slop/verification/20260923_run_mean_diff_cap_diagnostic.py'
ledger = root / 'outputs/bsbench-v2/costs.jsonl'
summary = root / 'outputs/bsbench-v2/run-summary.json'
cache = root / 'outputs/bsbench-v2/cache/calibration-candidates'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = {'ledger_sha256': sha(ledger), 'summary_sha256': sha(summary),
          'candidate_cache_files': sorted(path.name for path in cache.glob('*.json'))}
network_attempts = []


def deny_connection(self, address):
    network_attempts.append(str(address))
    raise AssertionError('no network requests are allowed in diagnostic --dry-run')


socket.socket.connect = deny_connection
socket.socket.connect_ex = deny_connection
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.argv = [str(script), '--dry-run']
output = io.StringIO()
with contextlib.redirect_stdout(output):
    runpy.run_path(str(script), run_name='__main__')
report = json.loads(output.getvalue())
after = {'ledger_sha256': sha(ledger), 'summary_sha256': sha(summary),
         'candidate_cache_files': sorted(path.name for path in cache.glob('*.json'))}
assert before == after
assert not network_attempts
assert 'modal' not in sys.modules
assert report['paid_dispatch'] is False
assert report['budget']['planned_gpu_jobs'] == 1 and report['budget']['estimated_judge_usd'] == 0
proof = {'author': 'PI/gpt-6-sol', 'network_attempts': network_attempts,
         'modal_imported': False, 'ledger_summary_candidate_cache_unchanged': True,
         'diagnostic_id': report['spec']['diagnostic_id'], 'budget': report['budget']}
path = root / 'slop/verification/20260923_mean_diff_cap_nodispatch_proof.json'
path.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
