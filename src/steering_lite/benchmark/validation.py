"""Local persona and judge-request validation for numbered BS-bench records."""
from __future__ import annotations

import hashlib
import json
import re

from .judge import request, request_key, score_pair

_REFUSAL = re.compile(r"\b(can(?:not|'t)|unable to|won't|refuse|cannot help)\b", re.I)


def validate_persona_examples(examples: list[dict]) -> list[dict]:
    """Reject trivial pair confounds before a paid validator is used."""
    checked = []
    for example in examples:
        pos, neg = example["positive"], example["negative"]
        labels = {example["positive_persona"].lower(), example["negative_persona"].lower()}
        lengths = (len(pos.split()), len(neg.split()))
        if not all(lengths) or max(lengths) / min(lengths) > 2:
            raise ValueError("persona example has confounded answer length")
        if _REFUSAL.search(pos) or _REFUSAL.search(neg):
            raise ValueError("persona example has refusal confound")
        if any(label in text.lower() for label in labels for text in (pos, neg)):
            raise ValueError("persona example copies a persona label into an answer")
        if not pos.endswith(example["shared_suffix"]) or not neg.endswith(example["shared_suffix"]):
            raise ValueError("persona pair no longer ends with its shared suffix")
        checked.append({"pair_id": example["pair_id"], "lengths": lengths, "status": "local_structural_checks"})
    return checked


def comparison_id(row: dict) -> str:
    """Opaque identity for one question, method, dose and steering side."""
    identity = [row["question_id"], row.get("method"), row.get("coefficient"), row["side"]]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:16]


def numbered_requests(rows: list[dict], model: str, endpoint: str) -> list[dict]:
    """Build persistable target-aware and target-blind requests without calling them."""
    records = []
    for row in rows:
        for order in ("AB", "BA"):
            for blind in (False, True):
                payload = request(row, order, model, blind=blind)
                if blind and {"method", "side", "coefficient"}.intersection(payload):
                    raise ValueError("blind request leaked intervention metadata")
                records.append({
                    "schema": "bsbench-judge-request-v1",
                    "question_id": row["question_id"], "question_number": row["question_number"],
                    "comparison_id": comparison_id(row), "order": order, "blind": blind,
                    "side": None if blind else row["side"],
                    "request_key": request_key(payload, endpoint), "payload": payload,
                })
    return records


def response_record(request_record: dict, response: dict) -> dict:
    """Schema-tag local/remote judge responses; blind output permits no change."""
    if request_record["blind"]:
        if not isinstance(response["changes"], list):
            raise ValueError("blind response changes must be a list")
    else:
        for key in ("on_axis_A", "on_axis_B", "off_axis_A", "off_axis_B"):
            if key not in response:
                raise ValueError(f"target-aware response missing {key}")
    return {key: request_record[key] for key in ("request_key", "question_id", "question_number", "comparison_id", "order", "blind", "side")} | {"schema": "bsbench-judge-response-v1", "response": response}


def disagreements(rows: list[dict]) -> list[dict]:
    """Keep AB/BA effects and blind descriptions separated by comparison."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["comparison_id"], row["question_id"]), []).append(row)
    result = []
    for (comparison, question_id), records in grouped.items():
        aware = [record for record in records if not record["blind"]]
        blind = [record for record in records if record["blind"]]
        effects = [score_pair(record["response"], record["order"], record["side"])["effect"] for record in aware]
        result.append({"comparison_id": comparison, "question_id": question_id, "target_effects": effects, "target_disagreement": max(effects) - min(effects) if effects else None, "blind_descriptions": [{"order": record["order"], "response": record["response"]} for record in blind]})
    return result


def persona_validation_request(example: dict, model: str) -> dict:
    """Build, but do not send, the later intended-behavior validation request."""
    prompt = f"""Compare two persona-pair completions for this scenario. Does stated intended behavior, rather than refusal, answer length, style, or persona echo, explain their difference?

Scenario: {example['scenario']}
Intended behavior: {example['intended_behavior']}
Positive persona: {example['positive_persona']}
Negative persona: {example['negative_persona']}
Positive completion: {example['positive']}
Negative completion: {example['negative']}

Return only JSON with intended_behavior_explains (boolean) and reason (one short sentence)."""
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_schema", "json_schema": {"name": "persona_pair_validation", "strict": True, "schema": {"type": "object", "properties": {"intended_behavior_explains": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["intended_behavior_explains", "reason"], "additionalProperties": False}}}}
