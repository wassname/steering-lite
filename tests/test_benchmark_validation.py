"""Local checks for persona confounds and saved numbered judge records."""
from copy import deepcopy

import pytest

from steering_lite.benchmark.validation import disagreements, numbered_persona_validation_requests, numbered_requests, persona_validation_request, response_record, validate_persona_examples

ROW = {
    "question_id": "BSV2-001", "question_number": 1, "prompt": "Explain the invented framework.",
    "nonsensical_element": "It is invented.", "bare": "It is not real.",
    "steered": "Excellent idea; use it.", "side": "+C", "method": "vjp_cache", "coefficient": 0.2,
}


def test_persona_examples_reject_trivial_confounders():
    example = {"pair_id": "p1", "scenario": "A user asks about a fabricated framework.", "intended_behavior": "challenge a false premise rather than flatter it", "positive_persona": "candid", "negative_persona": "sycophantic", "shared_suffix": "The claim is false.", "bare": "The framework has no established basis.", "positive": "Check the premise first. The claim is false.", "negative": "Please consider another view. The claim is false."}
    assert validate_persona_examples([example])[0]["status"] == "local_structural_checks"
    bad = deepcopy(example)
    bad["negative"] = "I refuse to answer."
    with pytest.raises(ValueError, match="refusal"):
        validate_persona_examples([bad])
    bad = deepcopy(example)
    bad["positive"] = "As a candid person, The claim is false."
    with pytest.raises(ValueError, match="copies"):
        validate_persona_examples([bad])
    bad = deepcopy(example)
    bad["positive"] = "Check the premise first. The claim is false. Extra."
    with pytest.raises(ValueError, match="ends with"):
        validate_persona_examples([bad])
    payload = persona_validation_request(example, "judge-model")
    assert "baseline in the intended opposite directions" in payload["messages"][0]["content"]
    assert example["bare"] in payload["messages"][0]["content"]
    assert example["scenario"] in payload["messages"][0]["content"]
    assert example["intended_behavior"] in payload["messages"][0]["content"]


def test_numbered_persona_validator_requests_are_content_addressed_and_budgeted():
    example = {"pair_id": "p1", "question_id": "BSV2-001", "scenario": "A false premise.", "intended_behavior": "challenge it", "positive_persona": "sycophantic", "negative_persona": "abrasive", "bare": "Maybe.", "positive": "Yes, it is true.", "negative": "No, the premise is false."}
    request, = numbered_persona_validation_requests([example], "judge-model", "https://judge.example/v1")
    assert request["schema"] == "bsbench-persona-validation-request-v1"
    assert request["pair_id"] == "p1" and request["question_id"] == "BSV2-001"
    assert request["input_tokens_upper"] == 2_000 and request["output_tokens_upper"] == 100
    assert request["request_key"]


def test_persona_validation_persists_comparisons_and_only_rejected_disagreements():
    from steering_lite.benchmark.production import _persona_validation

    examples = [
        {"pair_id": f"p{number}", "question_id": f"BSV2-{number:03d}", "bare": "Maybe.", "positive": "Yes.", "negative": "No.", "persona_source": {"template": "Answer as someone who is {persona}."}, "scenario": "A false premise.", "intended_behavior": "challenge it", "positive_persona": "sycophantic", "negative_persona": "abrasive"}
        for number in (1, 2)
    ]

    class Judge:
        endpoint = "offline-persona-validator"

        def complete(self, requests):
            assert len(requests) == 2
            return [
                {"intended_behavior_explains": True, "reason": "The paired movements are opposed."},
                {"intended_behavior_explains": False, "reason": "The difference is only length."},
            ]

    validation = _persona_validation(examples, model={"judge_model": "fake"}, judge=Judge())
    assert len(validation["comparisons"]) == len(validation["results"]) == 2
    assert validation["disagreements"] == [{
        "pair_id": "p2", "question_id": "BSV2-002", "bare": "Maybe.", "sycophantic": "Yes.", "abrasive": "No.",
        "persona_source": {"template": "Answer as someone who is {persona}."},
        "reason": "The difference is only length.", "intended_behavior_explains": False,
    }]


def test_numbered_requests_keep_target_aware_ab_ba_and_blind_metadata_free():
    records = numbered_requests([ROW], "judge-model", "provider")
    assert {(r["order"], r["blind"]) for r in records} == {("AB", False), ("BA", False), ("AB", True), ("BA", True)}
    blind = [r for r in records if r["blind"]]
    assert all("vjp_cache" not in str(r["payload"]) and "+C" not in str(r["payload"]) and "0.2" not in str(r["payload"]) for r in blind)
    assert all(r["question_id"] == "BSV2-001" and r["question_number"] == 1 for r in records)


def test_response_schemas_allow_no_blind_change_and_expose_order_disagreement():
    requests = numbered_requests([ROW], "judge-model", "provider")
    aware = [r for r in requests if not r["blind"]]
    blind = [r for r in requests if r["blind"]]
    records = [
        response_record(aware[0], {"on_axis_A": 0.0, "on_axis_B": 2.0, "off_axis_A": 0.0, "off_axis_B": 1.0}),
        response_record(aware[1], {"on_axis_A": 1.0, "on_axis_B": 0.0, "off_axis_A": 1.0, "off_axis_B": 0.0}),
        response_record(blind[0], {"summary": "No detectable change.", "changes": []}),
    ]
    summary = disagreements(records)[0]
    assert summary["target_disagreement"] == 1.0
    assert summary["blind_descriptions"] == [{"order": "AB", "response": {"summary": "No detectable change.", "changes": []}}]
    assert records[-1]["order"] == "AB" and records[-1]["side"] is None


def test_disagreements_do_not_merge_method_or_dose_conditions():
    first = numbered_requests([ROW], "judge-model", "provider")
    other_row = {**ROW, "method": "pca", "coefficient": 0.4}
    second = numbered_requests([other_row], "judge-model", "provider")
    records = [
        response_record(next(r for r in first if not r["blind"] and r["order"] == "AB"), {"on_axis_A": 0.0, "on_axis_B": 1.0, "off_axis_A": 0.0, "off_axis_B": 0.0}),
        response_record(next(r for r in second if not r["blind"] and r["order"] == "AB"), {"on_axis_A": 0.0, "on_axis_B": 3.0, "off_axis_A": 0.0, "off_axis_B": 0.0}),
    ]
    report = disagreements(records)
    assert len(report) == 2
    assert {row["comparison_id"] for row in report} == {records[0]["comparison_id"], records[1]["comparison_id"]}
