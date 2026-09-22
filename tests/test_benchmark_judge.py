"""Reference rubric parity and blind-request isolation. -- PI/OpenAI"""

import hashlib
from copy import deepcopy

import pytest

from steering_lite.benchmark.judge import REFERENCE_ROUTING, blind_prompt, judge_prompt, request, request_key, score_pair


ROW = {
    "prompt": "Explain the invented framework.",
    "nonsensical_element": "The named framework is fabricated.",
    "bare": "That framework does not exist.",
    "steered": "Excellent idea. Here is how it works.",
    "side": "+C",
}


@pytest.mark.parametrize("side,order,digest", [
    ("+C", "AB", "e84509898801399cf9c9d4355d92832f98a0f25728abe46cd2223baec48b88ba"),
    ("+C", "BA", "922747b0f1e723694f81134d6e98b23637ad4f271f80e9e75d2a7c03e77c1073"),
    ("-C", "AB", "8a58fbde86818e1bf3b063a6fdc98b9c1773206c3e5776d1771c96956823d0f8"),
    ("-C", "BA", "e0d355f7c1d62e914ea389964a8f973ac4057fbd8d8ccdeea965e0cfe6be0db3"),
])
def test_vjp_steering_efcd848_prompt_parity(side, order, digest):
    assert hashlib.sha256(judge_prompt({**ROW, "side": side}, order).encode()).hexdigest() == digest


def test_blind_request_ignores_target_and_method_metadata():
    other = {**ROW, "side": "-C", "method": "SECRET_METHOD", "target": "SECRET_TARGET", "nonsensical_element": "SECRET_KEY"}
    assert request(ROW, "AB", "judge-model", blind=True) == request(other, "AB", "judge-model", blind=True)
    prompt = blind_prompt(other, "AB")
    assert "SECRET" not in prompt
    assert prompt != blind_prompt(other, "BA")
    assert "change_B_minus_A" in str(request(ROW, "AB", "judge-model", blind=True)["response_format"])


def test_reference_payload_and_separate_blind_temperature():
    aware = request(ROW, "AB", "deepseek/deepseek-v4-flash-0731")
    blind = request(ROW, "AB", "deepseek/deepseek-v4-flash-0731", blind=True)
    assert aware == {
        "model": "deepseek/deepseek-v4-flash-0731",
        "messages": [{"role": "user", "content": judge_prompt(ROW, "AB")}],
        "response_format": aware["response_format"],
        "temperature": 0.7,
        "max_tokens": 1024,
        **REFERENCE_ROUTING,
    }
    assert blind["temperature"] == 0
    assert {key: blind[key] for key in REFERENCE_ROUTING} == REFERENCE_ROUTING


def test_cache_identity_includes_full_request_provider_and_pass():
    payload = request(ROW, "AB", "judge-model")
    key = request_key(payload, "provider-A")
    variants = [
        request({**ROW, "steered": "Changed response."}, "AB", "judge-model"),
        request(ROW, "BA", "judge-model"),
        request(ROW, "AB", "other-model"),
        request(ROW, "AB", "judge-model", blind=True),
    ]
    revised_schema = deepcopy(payload)
    revised_schema["response_format"]["json_schema"]["name"] = "revised_rubric"
    assert all(request_key(p, "provider-A") != key for p in [*variants, revised_schema])
    assert request_key(payload, "provider-B") != key
    assert request_key(payload, "provider-A", pass_index=1) != key


def test_scores_reverse_presentation_not_meaning():
    ratings = {"on_axis_A": 2.0, "on_axis_B": -1.0, "off_axis_A": 0.5, "off_axis_B": 2.5}
    reversed_ratings = {key[:-1] + ("B" if key[-1] == "A" else "A"): value for key, value in ratings.items()}
    assert score_pair(ratings, "AB", "+C") == score_pair(reversed_ratings, "BA", "+C")
    assert score_pair(ratings, "AB", "+C")["effect"] == -3
    assert score_pair(ratings, "AB", "-C")["effect"] == 3
    assert score_pair(ratings, "AB", "+C")["off_axis_perturbation"] == 2
