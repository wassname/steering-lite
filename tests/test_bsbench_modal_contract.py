from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from steering_lite.benchmark.pipeline import METHODS


@pytest.fixture
def modal_script():
    path = Path(__file__).parents[1] / "scripts" / "run_bsbench_modal.py"
    spec = importlib.util.spec_from_file_location("bsbench_modal_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", METHODS)
def test_modal_callback_serializes_each_method_without_dispatch(modal_script, method):
    calls = []
    modal_script.run_stage = SimpleNamespace(remote=lambda **kwargs: calls.append(kwargs) or {"offline": True})
    callback = modal_script.remote_stage_call("Qwen/Qwen3.5-4B", explicit_run=True, budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0})
    stage = "generation" if method in {"bare", "prompting"} else "calibration-candidates"
    result = callback(stage=stage, method=method, config={"serializable": [1, 2]}, prompts=["prompt"])
    assert result == {"offline": True}
    assert calls == [{
        "stage": stage,
        "method": method,
        "config": {"serializable": [1, 2]},
        "prompts": ["prompt"],
        "model_id": "Qwen/Qwen3.5-4B",
    }]


def test_modal_callback_refuses_dispatch_without_explicit_run(modal_script):
    calls = []
    modal_script.run_stage = SimpleNamespace(remote=lambda **kwargs: calls.append(kwargs))
    callback = modal_script.remote_stage_call("Qwen/Qwen3.5-4B", explicit_run=False, budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0})
    with pytest.raises(RuntimeError, match="explicit --run"):
        callback(stage="generation", method="bare", config={}, prompts=[])
    assert not calls


def test_candidate_policy_records_coherence_failure_or_search_limit(modal_script):
    class Vector:
        @contextmanager
        def __call__(self, _model, *, C):
            yield C

    def generate(_model, _tokenizer, prompts, _batch_size, _max_new_tokens):
        return ["answer." for _ in prompts]

    coefficients, items, health, search = modal_script._candidate_policy(
        Vector(), object(), object(), ["one", "two"], limit=2, max_new_tokens=8,
        generate=generate, health=lambda _tokenizer, _answers: ({"answers": 2}, []),
    )
    assert coefficients == [0.1, 0.2]
    assert len(items) == 4 and set(health) == {"0.1", "0.2"}
    assert search["termination"] == "search_limit" and len(search["history"]) == 2

    _, _, _, failed = modal_script._candidate_policy(
        Vector(), object(), object(), ["one"], limit=2, max_new_tokens=8,
        generate=generate, health=lambda _tokenizer, _answers: ({"answers": 1}, ["repetition"]),
    )
    assert failed["termination"] == "coherence_failure" and len(failed["history"]) == 1
