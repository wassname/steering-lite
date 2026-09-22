"""Fail closed before any provider call or ledger reservation in the exact paid CLI path. — PI/gpt-6-sol"""
import atexit
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

if os.environ.get('BSBENCH_DIAGNOSTIC_IMPORT_PROOF') == '1':
    root = Path(__file__).resolve().parents[3]
    ledger = root / 'outputs/bsbench-v2/costs.jsonl'
    result = root / 'slop/verification/20260923_mean_diff_cap_import_proof.json'
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    calls = {'provider_read': 0, 'socket': 0, 'reservation': 0}
    assert Path(sys.argv[0]).name == '20260923_run_mean_diff_cap_diagnostic.py'
    assert str(root) in sys.path

    import modal
    import steering_lite.benchmark.cache as benchmark_cache

    def deny_provider_read(self, path):
        calls['provider_read'] += 1
        raise RuntimeError('BSBENCH_IMPORT_PROOF_DENIED_PROVIDER_READ')

    def deny_socket(self, address):
        calls['socket'] += 1
        raise RuntimeError('BSBENCH_IMPORT_PROOF_DENIED_NETWORK')

    def deny_reservation(*args, **kwargs):
        calls['reservation'] += 1
        raise RuntimeError('BSBENCH_IMPORT_PROOF_DENIED_RESERVATION')

    modal.Volume.read_file = deny_provider_read
    socket.socket.connect = deny_socket
    socket.socket.connect_ex = deny_socket
    benchmark_cache.reserve_many = deny_reservation

    def record():
        after = hashlib.sha256(ledger.read_bytes()).hexdigest()
        result.write_text(json.dumps({
            'author': 'PI/gpt-6-sol', 'argv': sys.argv,
            'repo_root_on_sys_path': str(root) in sys.path,
            'callback_import_reached_provider_read': calls['provider_read'] == 1,
            'provider_read_attempts_denied': calls['provider_read'],
            'socket_attempts_denied': calls['socket'],
            'reservation_attempts_denied': calls['reservation'],
            'ledger_sha256_before': before, 'ledger_sha256_after': after,
            'ledger_unchanged': before == after,
        }, indent=2) + '\n')

    atexit.register(record)
