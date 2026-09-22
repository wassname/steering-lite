from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.production import _candidate_items
from steering_lite.benchmark.sweep import GPU_HOURS_PER_STAGE, MODAL_GPU_STAGE_TIMEOUT_SECONDS


@pytest.fixture
def modal_script():
    path = Path(__file__).parents[1] / "scripts" / "run_bsbench_modal.py"
    spec = importlib.util.spec_from_file_location("bsbench_modal_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_modal_timeout_and_reservation_hours_share_one_constant(modal_script):
    assert modal_script.MODAL_GPU_STAGE_TIMEOUT_SECONDS == MODAL_GPU_STAGE_TIMEOUT_SECONDS == 45 * 60
    assert GPU_HOURS_PER_STAGE == MODAL_GPU_STAGE_TIMEOUT_SECONDS / 3600


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


def test_canonical_prompt_serializer_keeps_generation_and_kl_inputs_equivalent(modal_script):
    class Tokenizer:
        def __init__(self):
            self.chat_calls = []
            self.token_calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.chat_calls.append((messages, kwargs))
            return f"thinking={kwargs['enable_thinking']}|{messages[0]['content']}"

        def __call__(self, text, **kwargs):
            self.token_calls.append((text, kwargs))
            return SimpleNamespace(input_ids=[[text]])

    tokenizer = Tokenizer()
    spec = {"template": "Answer in 2 short sentences.", "enable_thinking": False}
    generated = modal_script.canonical_prompt_texts(tokenizer, ["Question"], spec)
    measured = modal_script.canonical_prompt_ids(tokenizer, ["Question"], spec)
    assert generated == ["thinking=False|Question Answer in 2 short sentences."]
    assert measured == [[generated[0]]]
    assert all(call[1]["enable_thinking"] is False for call in tokenizer.chat_calls)
    assert tokenizer.token_calls == [(generated[0], {"add_special_tokens": False, "return_tensors": "pt"})]


def test_pinned_qwen_layers(modal_script):
    model = type("Model", (), {"config": type("Config", (), {"layer_types": ["linear_attention"] * 32})()})()
    for layer in (7, 11, 15, 19, 23):
        model.config.layer_types[layer] = "full_attention"
    assert modal_script._layers(model) == ((7, 11, 15, 19, 23), 29)


def test_candidate_policy_records_coherence_failure_or_search_limit(modal_script):
    coefficients_seen = []
    class Vector:
        @contextmanager
        def __call__(self, _model, *, C):
            coefficients_seen.append(C)
            yield C

    def generate(_model, _tokenizer, prompts, _batch_size, _max_new_tokens):
        return ["answer." for _ in prompts]

    source_prompts = ["raw one", "raw two"]
    source_hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in source_prompts]
    coefficients, items, health, search = modal_script._candidate_policy(
        Vector(), object(), object(), ["serialized one", "serialized two"], prompt_sha256s=source_hashes, limit=2, max_new_tokens=8,
        generate=generate, health=lambda _tokenizer, _answers: ({"answers": 2}, []),
    )
    assert coefficients == [0.1, 0.2]
    assert len(items) == 8 and {item["prompt_sha256"] for item in items} == set(source_hashes) and set(health) == {"0.1:+C", "0.1:-C", "0.2:+C", "0.2:-C"}
    assert coefficients_seen == [0.1, -0.1, 0.2, -0.2]
    assert _candidate_items(coefficients, source_prompts, items) == items
    assert search["termination"] == "search_limit" and len(search["history"]) == 4

    _, _, _, failed = modal_script._candidate_policy(
        Vector(), object(), object(), ["serialized one"], prompt_sha256s=["raw-one"], limit=2, max_new_tokens=8,
        generate=generate, health=lambda _tokenizer, _answers: ({"answers": 1}, ["repetition"]),
    )
    assert failed["termination"] == "coherence_failure" and len(failed["history"]) == 2

    with pytest.raises(ValueError, match="source hashes"):
        modal_script._candidate_policy(
            Vector(), object(), object(), ["serialized one"], prompt_sha256s=[], limit=2, max_new_tokens=8,
            generate=generate, health=lambda _tokenizer, _answers: ({"answers": 1}, []),
        )
