"""Recompute finished mean_diff and historical selections without dispatch. — PI/gpt-6-sol"""

import collections
import hashlib
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

from steering_lite.benchmark.adapters import judge_cache_identity, judge_request_cached
from steering_lite.benchmark.cache import content_key, source_hash
from steering_lite.benchmark.judge import score_pair, validate_judgment
from steering_lite.benchmark.sweep import load_judge_pricing
from steering_lite.benchmark.validation import numbered_requests, response_record

ROOT = Path("outputs/bsbench-v2-mean-diff-cap384")
OLD = Path("outputs/bsbench-v2/results")
LEDGER = Path("outputs/bsbench-v2/costs.jsonl")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pricing = Path("slop/verification/20260922_v4-provider-endpoint-metadata.json")
    load_judge_pricing(pricing)
    generation_path, = (ROOT / "cache/final-generation").glob("*.json")
    judgments_path, = (ROOT / "cache/final-judgments").glob("*.json")
    generation = read(generation_path)
    judgment = read(judgments_path)
    summary_path = ROOT / "run-summary.json"
    summary = read(summary_path)
    condition = summary["conditions"]["mean_diff"]
    final, saved = generation["result"], judgment["result"]
    assert source_hash() == generation["identity"]["code_sha256"] == judgment["identity"]["code_sha256"]
    assert summary["methods"] == ["mean_diff"] and condition["final_judgments"] == saved
    assert {k: v for k, v in condition["final"].items() if k != "reused"} == final
    assert content_key(summary["identity"]) == summary["identity_sha256"]
    plan = final["executable_generation_plan"]
    assert len(plan) == len(final["answers"]) == len(final["health_records"]) == 168
    assert final["plan_sha256"] == content_key({"plan": plan})
    records = {r["prompt_id"]: r for case in generation["identity"]["config"]["transfer_prompt_records"].values() for r in case}
    assert len(records) == 28 and set(records) == set(final["baseline_answers"])
    rows = []
    health_by_dose = collections.defaultdict(list)
    for item, answer, health in zip(plan, final["answers"], final["health_records"], strict=True):
        assert all(item[k] == health[k] for k in ("case_id", "prompt_id", "magnitude", "side"))
        if item["case_id"] != "bsbench-v2-evaluation":
            continue
        source = records[item["prompt_id"]]
        assert item["prompt"] == source["prompt"] and item["prompt_sha256"] == source["content_sha256"]
        rows.append({"question_id": item["prompt_id"], "question_number": int(item["prompt_id"].split("-")[-1]),
                     "prompt": source["prompt"], "nonsensical_element": source["answer_key"],
                     "bare": final["baseline_answers"][item["prompt_id"]], "steered": answer,
                     "method": "mean_diff", "magnitude": item["magnitude"], "random_seed": 0,
                     "side": item["side"], "generation_health": health})
        health_by_dose[item["side"], item["magnitude"]].append(health)
    assert len(rows) == 120
    expected = numbered_requests(rows, summary["identity"]["model"]["judge_model"], ENDPOINT)
    assert saved["requests"] == expected and len(expected) == len(saved["responses"]) == 720
    assert saved["aware"] == [r for r in saved["responses"] if not r["blind"]]
    assert saved["blind"] == [r for r in saved["responses"] if r["blind"]]
    assert len(saved["aware"]) == 480 and len(saved["blind"]) == 240
    original = read(Path("slop/verification/20260923_mean_diff_final_failure_cache_audit.json"))["requests"]
    assert len(original) == 720
    matches = collections.defaultdict(list)
    keyed = collections.defaultdict(list)
    for request, response, prior in zip(expected, saved["responses"], original, strict=True):
        assert all(request[k] == response[k] for k in ("request_key", "question_id", "question_number", "comparison_id", "order", "pass", "blind", "side", "magnitude", "random_seed"))
        content = {k: v for k, v in response["response"].items() if k not in ("_remote_usage", "_remote_cost_usd")}
        validate_judgment(content, request["payload"]["response_format"]["json_schema"]["schema"])
        identity = judge_cache_identity(request, ENDPOINT)
        key = content_key(identity)
        assert key == prior["key"] and identity["evidence_id"] == prior["evidence_id"]
        assert judge_request_cached(ROOT, request, ENDPOINT)
        record = read(ROOT / "cache/judge-request" / f"{key}.json")
        assert record["identity"] == identity and record["result"]["response"] == response["response"]
        assert response_record(request, record["result"]["response"]) == response
        matches[response["comparison_id"]].append(response)
        if not response["blind"]:
            keyed[response["side"], response["magnitude"]].append(response)
    assert len(matches) == 120 and all(len(v) == 6 for v in matches.values())
    assert all({(r["order"], r["pass"], r["blind"]) for r in v} ==
               {(o, p, b) for o in ("AB", "BA") for b, passes in ((False, (0, 1)), (True, (0,))) for p in passes}
               for v in matches.values())
    points = []
    for (side, magnitude), responses in sorted(keyed.items()):
        health = health_by_dose[side, magnitude]
        aware = [r for r in responses if not r["blind"]]
        assert len(health) == 20 and len(aware) == len(responses) == 80
        effects = [score_pair(r["response"], r["order"], side) for r in aware]
        on = statistics.mean(r["directed_intended_effect"] for r in effects)
        off = statistics.mean(abs(r["off_axis_perturbation"]) for r in effects)
        totals = {key: sum(r["metrics"][key] for r in health) for key in ("unfinished", "role_leaks", "repeated")}
        eligible = totals["unfinished"] / 20 < .5 and totals["role_leaks"] / 20 < .25 and totals["repeated"] / 20 < .25
        multiplier, = {x["multiplier"] for x in plan if x["case_id"] == "bsbench-v2-evaluation" and x["side"] == side and x["magnitude"] == magnitude}
        points.append({"method": "mean_diff_cap384", "random_seed": 0, "side": side, "multiplier": multiplier,
                       "magnitude": magnitude, "dose_score": on - 4 * off, "directed_intended_effect": on,
                       "absolute_off_axis_change": off, "cohort_eligible": eligible,
                       "health_flags": sum(bool(r["reasons"]) for r in health), "health_totals": totals,
                       "aware_count": len(aware), "blind_count": len(responses) - len(aware)})
    assert len(points) == 6
    selected_new = {side: max((p for p in points if p["side"] == side and p["cohort_eligible"]), key=lambda p: (p["dose_score"], p["directed_intended_effect"])) for side in ("+C", "-C")}
    historic = read(OLD / "measured-points.json")
    assert content_key(historic["points"]) == historic["points_sha256"]
    controls = []
    for side in ("+C", "-C"):
        for method in ("random", "mean_diff", "pca", "kv_cache_gram", "vjp_delta", "vjp_cache"):
            for seed in (range(5) if method == "random" else (0,)):
                options = [r for r in historic["points"] if r["method"] == method and r["side"] == side and r["random_seed"] == seed and r["cohort_eligible"]]
                best = max(options, key=lambda r: (r["dose_score"], r["directed_intended_effect"]))
                controls.append({k: best[k] for k in ("method", "random_seed", "side", "multiplier", "magnitude", "dose_score", "directed_intended_effect", "absolute_off_axis_change", "health_flags", "cohort_eligible", "point_id", "raw_evidence")})
    ledger_rows = [json.loads(line) for line in LEDGER.read_text().splitlines()]
    new_reservations = [r for r in ledger_rows if r["event"] == "reserved" and r["time"] >= "2026-09-23T03:00:00+00:00"]
    assert len(new_reservations) == 337 and all(r["kind"].startswith("judge-") for r in new_reservations)
    settled = {r["reservation"]: r for r in ledger_rows if r["event"] == "settled"}
    estimated = {r["reservation"]: r for r in ledger_rows if r["event"] == "estimated_at_reservation_upper"}
    assert sum(r["id"] in settled for r in new_reservations) == 333
    assert sum(r["id"] in estimated for r in new_reservations) == 4
    prior_states = {r["evidence_id"]: r for r in original if not r["cached"]}
    new_attempts = collections.Counter()
    for r in new_reservations:
        evidence_id = r["kind"].split("-")[1]
        assert evidence_id in prior_states
        new_attempts[evidence_id] += 1
    assert len(new_attempts) == 333 and max(new_attempts.values()) <= 3
    reservation_by_id = {r["id"]: r for r in ledger_rows if r["event"] == "reserved"}
    for state in original:
        record = read(ROOT / "cache/judge-request" / f"{state['key']}.json")["result"]
        times = [reservation_by_id[id]["time"] for id in record["attempt_reservations"]]
        assert all((time < "2026-09-23T03:00:00+00:00") == state["cached"] for time in times)
    retry_receipts = [estimated[r["id"]]["receipt"] for r in new_reservations if r["id"] in estimated]
    assert len(retry_receipts) == 4 and all(r["exception_type"] == "HTTPError" and r["status"] == 504 for r in retry_receipts)
    assert all(Path(r["provider_evidence"]).is_file() for r in retry_receipts)
    actual = sum((Decimal(str(settled[r["id"]]["actual_usd"] if r["id"] in settled else r["upper_usd"])) for r in new_reservations), Decimal(0))
    upper = sum((Decimal(str(r["upper_usd"])) for r in new_reservations), Decimal(0))
    prior_failed = read(Path("slop/verification/20260923_mean_diff_final_failure_ledger_audit.json"))
    before = Decimal(prior_failed["committed_ledger_usd"])
    current = sum((Decimal(str(settled[r["id"]]["actual_usd"] if r["id"] in settled else r["upper_usd"])) for r in ledger_rows if r["event"] == "reserved"), Decimal(0))
    assert abs((current - before) - actual) < Decimal("0.000000001")
    out = {"author": "PI/gpt-6-sol", "source_sha256": source_hash(), "summary": str(summary_path), "summary_sha256": sha(summary_path),
           "final_generation": str(generation_path), "final_generation_sha256": sha(generation_path),
           "final_judgments": str(judgments_path), "final_judgments_sha256": sha(judgments_path),
           "historical_points": str(OLD / "measured-points.json"), "historical_points_sha256": sha(OLD / "measured-points.json"),
           "requests": 720, "validated_cache_requests": 720, "original_cache_reuse": 387, "newly_completed_requests": 333,
           "aware": 480, "blind": 240, "matched_comparisons": 120, "points": points,
           "selected_repaired": selected_new, "selected_historical": controls,
           "resume_reservations": len(new_reservations), "resume_retries": len(new_reservations) - len(new_attempts),
           "resume_settled": 333, "resume_estimated_at_upper": 4, "resume_failure_types": {"OpenRouter/OpenInference HTTP504": 4},
           "resume_failure_evidence_paths": [r["provider_evidence"] for r in retry_receipts],
           "resume_actual_plus_upper_usd": str(actual),
           "resume_reservation_upper_usd": str(upper), "committed_ledger_usd": str(current),
           "external_phase6_plus_ancillary_hold_usd": "3", "all_held_usd": str(current + Decimal(3)),
           "gpu_reservations_on_resume": 0}
    path = Path("slop/verification/20260923_mean_diff_final_comparison.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("validated", len(saved["responses"]), "responses and", len(matches), "comparisons; no GPU reservation; actual+upper", actual)
    for side in ("+C", "-C"):
        new = selected_new[side]
        random = [p for p in controls if p["method"] == "random" and p["side"] == side]
        print(side, "new", new["dose_score"], "old", next(p["dose_score"] for p in controls if p["method"] == "mean_diff" and p["side"] == side), "random", [p["dose_score"] for p in random])


if __name__ == "__main__":
    main()
