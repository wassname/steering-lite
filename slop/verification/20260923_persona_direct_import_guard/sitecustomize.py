"""Deny every provider/reservation route during the exact direct-CLI bootstrap proof. — PI/gpt-6-sol"""
import atexit
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

if os.environ.get('BSBENCH_PERSONA_DIRECT_IMPORT_PROOF') == '1':
    root = Path(__file__).resolve().parents[3]
    ledger = root / 'outputs/bsbench-v2/costs.jsonl'
    output = root / 'slop/verification/20260923_persona_direct_import_proof.json'
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    root_initially_missing = str(root) not in sys.path
    calls = {'provider_read': 0, 'socket': 0, 'reservation': 0, 'app_run': 0}
    assert Path(sys.argv[0]).name == '20260923_run_persona_direct_diagnostic.py'
    assert root_initially_missing, 'proof must test direct CLI without PYTHONPATH repo root'

    import modal
    import steering_lite.benchmark.cache as benchmark_cache

    def deny_provider_read(self, path):
        calls['provider_read'] += 1
        raise RuntimeError('BSBENCH_PERSONA_PROOF_DENIED_PROVIDER_READ')

    def deny_socket(self, address):
        calls['socket'] += 1
        raise RuntimeError('BSBENCH_PERSONA_PROOF_DENIED_NETWORK')

    def deny_reservation(*args, **kwargs):
        calls['reservation'] += 1
        raise RuntimeError('BSBENCH_PERSONA_PROOF_DENIED_RESERVATION')

    def deny_app_run(*args, **kwargs):
        calls['app_run'] += 1
        raise RuntimeError('BSBENCH_PERSONA_PROOF_DENIED_APP_RUN')

    modal.Volume.read_file = deny_provider_read
    modal.App.run = deny_app_run
    socket.socket.connect = deny_socket
    socket.socket.connect_ex = deny_socket
    benchmark_cache.reserve_many = deny_reservation

    def record():
        after = hashlib.sha256(ledger.read_bytes()).hexdigest()
        output.write_text(json.dumps({
            'author': 'PI/gpt-6-sol', 'argv': sys.argv,
            'root_initially_missing': root_initially_missing,
            'repo_root_bootstrapped': str(root) in sys.path,
            'callback_import_reached_provider_read': calls['provider_read'] == 1,
            'denied_calls': calls, 'ledger_sha256_before': before,
            'ledger_sha256_after': after, 'ledger_unchanged': before == after,
        }, indent=2) + '\n')

    atexit.register(record)
