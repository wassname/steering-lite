"""Fail-closed no-dispatch guard for one real sweep CLI replay. — PI/OpenAI"""
import os
import sys
from pathlib import Path

if Path(sys.argv[0]).name == 'run_bsbench_sweep.py':
    try:
        import atexit
        import json
        import threading
        from dotenv import find_dotenv, load_dotenv

        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / 'scripts'))
        from steering_lite.benchmark import adapters, production
        import run_bsbench_modal

        counts = {'gpu': 0, 'judge': 0, 'reservation': 0}
        lock = threading.Lock()
        def forbidden(kind):
            def call(*args, **kwargs):
                with lock:
                    counts[kind] += 1
                raise AssertionError(f'CLI replay attempted {kind}')
            return call
        def modal_factory(*args, **kwargs):
            return forbidden('gpu')
        def judge_factory(*args, **kwargs):
            return forbidden('judge')
        adapters.openrouter_request_callback = judge_factory
        run_bsbench_modal.remote_stage_call = modal_factory
        adapters.reserve = forbidden('reservation')
        production.reserve = forbidden('reservation')
        load_dotenv(find_dotenv(usecwd=True))
        assert os.environ['OPENROUTER_API_KEY']
        assert not (root / '.local/bsbench-cli-proof/no-shell-env').exists()
        evidence = root / 'slop/verification/20260922_actual-just-sweep'
        def record_exit():
            Path(str(evidence) + '-counts.json').write_text(json.dumps({'author':'PI/OpenAI','pid':os.getpid(),'armed':True,'callback_attempts':counts},indent=2)+'\n')
        atexit.register(record_exit)
        Path(str(evidence) + '-armed.json').write_text(json.dumps({'author':'PI/OpenAI','pid':os.getpid(),'guard_module':str(Path(__file__).resolve()),'armed':True,'dotenv_loaded_without_printing':True},indent=2)+'\n')
        print('CLI replay guards armed; outgoing calls and reservations will fail.',flush=True)
    except BaseException as error:
        sys.stderr.write(f'CLI replay guard setup failed: {type(error).__name__}\n')
        sys.stderr.flush()
        os._exit(99)
