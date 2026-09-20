"""Write the non-paying BS-bench preflight or import a completed production stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from steering_lite.benchmark.adapters import openrouter_request_callback, real_adapters
from steering_lite.benchmark.cache import committed, content_key, save_json, settle_receipt, source_hash
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.production import record_completed_stage, run_condition, run_stages
from steering_lite.benchmark.sweep import JUDGE_MODEL, MODEL_ID, dry_manifest


def run_full_sweep(root: Path, ledger: Path, *, model: dict, rows: list[dict], backend, prompt_spec: dict, judge, methods: tuple[str, ...] = METHODS, measure=None, solver=None, vector_loader=None, transfer_records=None) -> dict:
    """Run named conditions in order and atomically save only a matching run summary."""
    summary_path = root / "run-summary.json"
    identity = {
        "schema": "bsbench-run-summary-identity-v1",
        "model": model,
        "cohort": cohort_identity(rows),
        "prompt_spec": prompt_spec,
        "judge": {"model": model["judge_model"], "endpoint": judge.endpoint},
        "methods": list(methods),
        "code_sha256": content_key({
            "benchmark": source_hash(),
            "entrypoint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }),
    }
    prior = json.loads(summary_path.read_text()) if summary_path.exists() else None
    conditions = dict(prior["conditions"]) if prior and prior.get("identity") == identity else {}
    for method in methods:
        conditions[method] = run_condition(
            root,
            ledger,
            model=model,
            data=cohort_identity(rows),
            method=method,
            rows=rows,
            backend=backend,
            prompt_spec=prompt_spec,
            judge=judge,
            measure=measure,
            solver=solver,
            vector_loader=vector_loader,
            transfer_records=transfer_records,
        )
        save_json(summary_path, {"schema": "bsbench-run-summary-v1", "identity": identity, "identity_sha256": content_key(identity), "methods": list(methods), "conditions": conditions})
    return {"schema": "bsbench-run-summary-v1", "identity": identity, "identity_sha256": content_key(identity), "methods": list(methods), "conditions": conditions, "summary_path": str(summary_path), "paid_execution_enabled": bool(getattr(backend, "paid_execution_enabled", False))}


def _redact(value):
    if isinstance(value, dict):
        return {
            key: "<redacted>" if any(word in key.lower() for word in ("key", "token", "secret", "authorization", "hash", "user_id")) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _provider_headers(headers) -> dict:
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in {"date", "content-type", "content-length", "retry-after", "x-request-id", "cf-ray"}
        or key.lower().startswith("x-ratelimit-")
    }


def _request_identity(payload: dict) -> dict:
    return {
        "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model": payload["model"],
        "response_schema": payload["response_format"]["json_schema"]["name"],
    }


@contextmanager
def _openrouter_read_timeout(seconds: float):
    """Apply the fixed read timeout without changing production-cache source identity."""
    if seconds <= 0:
        raise ValueError("OpenRouter read timeout must be positive")
    import steering_lite.benchmark.adapters as adapters

    original_urlopen = adapters.urlopen

    def timed_urlopen(request, *args, **kwargs):
        return original_urlopen(request, *args, **(kwargs | {"timeout": seconds}))

    adapters.urlopen = timed_urlopen
    try:
        yield
    finally:
        adapters.urlopen = original_urlopen


def audited_openrouter_request_callback(*, endpoint: str, api_key: str, evidence_root: Path, min_interval_seconds: float = 10.0):
    request_call = openrouter_request_callback(endpoint=endpoint, api_key=api_key)
    next_request_at = 0.0
    attempt = 0

    def call(payload: dict) -> dict:
        nonlocal attempt, next_request_at
        attempt += 1
        wait_seconds = max(next_request_at - time.monotonic(), 0.0)
        if wait_seconds:
            time.sleep(wait_seconds)
        dispatch_started_monotonic = time.monotonic()
        dispatch_started_at = datetime.now(timezone.utc).isoformat()
        next_request_at = dispatch_started_monotonic + min_interval_seconds
        identity = _request_identity(payload)

        def timing(*, schema: str, outcome: str) -> dict:
            response_finished_monotonic = time.monotonic()
            response_finished_at = datetime.now(timezone.utc).isoformat()
            return {
                "schema": schema,
                "recorded_at": response_finished_at,
                "endpoint": endpoint,
                **identity,
                "attempt": attempt,
                "outcome": outcome,
                "dispatch_started_at": dispatch_started_at,
                "response_finished_at": response_finished_at,
                "elapsed_seconds": response_finished_monotonic - dispatch_started_monotonic,
                "enforced_wait_seconds": wait_seconds,
                "min_interval_seconds": min_interval_seconds,
            }

        def persist(evidence: dict) -> None:
            save_json(evidence_root / f"{identity['payload_sha256']}-{attempt:06d}-{evidence['outcome']}.json", evidence)

        try:
            response = request_call(payload)
        except HTTPError as error:
            raw_body = error.read().decode(errors="replace")
            try:
                body = _redact(json.loads(raw_body))
            except json.JSONDecodeError:
                body = {"raw_body": raw_body[:10_000]}
            evidence = timing(schema="bsbench-openrouter-http-error-v1", outcome="failed") | {
                "status": error.code,
                "reason": error.reason,
                "headers": _provider_headers(error.headers),
                "body": body,
            }
            persist(evidence)
            raise
        except (TimeoutError, URLError, RemoteDisconnected, ConnectionResetError) as error:
            evidence = timing(schema="bsbench-openrouter-no-response-v1", outcome="failed") | {
                "exception_type": type(error).__name__,
            }
            persist(evidence)
            raise
        except Exception as error:
            evidence = timing(schema="bsbench-openrouter-callback-error-v1", outcome="failed") | {
                "exception_type": type(error).__name__,
            }
            if hasattr(error, "evidence"):
                evidence["parse_evidence"] = _redact(error.evidence)
            persist(evidence)
            raise
        else:
            persist(timing(schema="bsbench-openrouter-response-timing-v1", outcome="success"))
            return response

    return call


def openrouter_metadata(*, api_key: str) -> dict:
    def get(resource: str) -> dict:
        request = Request(f"https://openrouter.ai/api/v1/{resource}", headers={"Authorization": f"Bearer {api_key}"}, method="GET")
        try:
            with urlopen(request, timeout=30) as response:
                return {"status": response.status, "headers": _provider_headers(response.headers), "body": _redact(json.loads(response.read()))}
        except HTTPError as error:
            raw_body = error.read().decode(errors="replace")
            try:
                body = _redact(json.loads(raw_body))
            except json.JSONDecodeError:
                body = {"raw_body": raw_body[:10_000]}
            return {"status": error.code, "reason": error.reason, "headers": _provider_headers(error.headers), "body": body}

    return {
        "schema": "bsbench-openrouter-nongeneration-metadata-v1",
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": {"key": get("key"), "credits": get("credits")},
    }


def import_recorded_smoke(out: Path, model_id: str) -> dict:
    """Import the already-completed Modal smoke; do not dispatch another GPU job."""
    summary = json.loads(Path("slop/verification/20260919_phase6-modal-smoke-summary.json").read_text())
    if summary["model"] != model_id:
        raise ValueError(f"recorded smoke model {summary['model']!r} does not match {model_id!r}")
    rows = read_dev_cohort()[:1]
    if summary["question_id"] != rows[0]["question_id"]:
        raise ValueError("recorded smoke question does not match the fixed development cohort")
    model = {"id": model_id}
    data = cohort_identity(rows)
    prompts = [rows[0]["prompt"]]
    records = []
    for method in summary["conditions"]:
        records.append(record_completed_stage(
            out,
            stage="generation",
            model=model,
            data=data,
            method=method,
            config={"recorded_modal_app": summary["modal_app"], "question_id": summary["question_id"]},
            prompts=prompts,
            result=summary["generation_records"][method] | {"recorded_modal_smoke": summary, "method": method},
        ))
    return {"mode": "run-recorded-smoke", "records": records, "production_cache": str(out / "cache")}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--import-smoke", action="store_true")
    mode.add_argument("--import-receipt", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check-openrouter-env", action="store_true")
    mode.add_argument("--openrouter-metadata", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("outputs/bsbench-v2"))
    parser.add_argument("--method", "--stage", dest="method", choices=METHODS, help="Run one condition for recovery or debugging; omit for the canonical full sweep.")
    parser.add_argument("--backend", choices=("recorded", "fake", "real"))
    parser.add_argument("--judge-endpoint", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--openrouter-read-timeout", type=float, default=180.0)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--metadata-out", type=Path)
    args = parser.parse_args()
    if args.dry_run:
        result = dry_manifest(args.out, args.model, cache_aware=True)
    elif args.check_openrouter_env:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        result = {"mode": "check-openrouter-env", "openrouter_api_key_present": True}
    elif args.openrouter_metadata:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        evidence = openrouter_metadata(api_key=api_key)
        path = args.metadata_out or args.out / "provider-evidence" / f"openrouter-metadata-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        save_json(path, evidence)
        result = {"mode": "openrouter-metadata", "evidence_path": str(path), "statuses": {name: endpoint["status"] for name, endpoint in evidence["endpoints"].items()}}
    elif args.import_smoke:
        result = import_recorded_smoke(args.out, args.model)
    elif args.import_receipt:
        receipt = json.loads(args.import_receipt.read_text())
        if set(receipt) != {"reservation", "actual_usd", "receipt"}:
            raise ValueError("receipt JSON must contain reservation, actual_usd and receipt")
        ledger = args.ledger or args.out / "costs.jsonl"
        settle_receipt(ledger, receipt["reservation"], receipt["actual_usd"], receipt["receipt"])
        result = {"mode": "receipt-import", "ledger": str(ledger), "reservation": receipt["reservation"], "actual_usd": receipt["actual_usd"]}
    else:
        if args.backend == "fake":
            rows = read_dev_cohort()
            class FakeBackend:
                """Deterministic offline remote-contract backend; its records are not experiment results."""
                remote_vector_binding = True

                def __init__(self): self.calls = []

                def gpu(self, *, stage, method, config, prompts):
                    self.calls.append(stage)
                    if stage == "generation":
                        result = {"actual_usd": 0.0, "answers": [f"Fake {method} answer." for _ in prompts], "health_records": [{"question_id": prompt_id, "reasons": []} for prompt_id in config["prompt_ids"]]}
                        if method == "prompting" and "persona_validation_prompt_ids" in config:
                            result["persona_validation_pairs"] = [
                                {"question_id": prompt_id, "sycophantic": "Fake agreement.", "abrasive": "Fake challenge."}
                                for prompt_id in config["persona_validation_prompt_ids"]
                            ]
                        return result
                    if stage == "calibration-candidates":
                        coefficients = [0.2, 0.4]
                        return {"actual_usd": 0.0, "vector_bytes": b"offline-fake-vector-v1", "baseline_answers": ["Fake baseline." for _ in prompts], "candidate_coefficients": coefficients, "candidate_health": {str(coefficient): {"reasons": []} for coefficient in coefficients}, "candidate_items": [{"coefficient": coefficient, "prompt_index": index, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(), "response": f"Fake candidate {coefficient}/{index}."} for coefficient in coefficients for index, prompt in enumerate(prompts)], "method_config": {"fake": True}}
                    if stage == "final-generation":
                        from steering_lite.benchmark.cache import content_key
                        from steering_lite.benchmark.dose_search import PREDICTION_CASES, final_dose_plan
                        from steering_lite.benchmark.transfer_data import load_evaluation_records, load_transfer_records

                        artifact = config["vector_artifact"]
                        assert __import__("hashlib").sha256(__import__("base64").b64decode(artifact["vector_bytes_b64"])).hexdigest() == artifact["sha256"]
                        target = {"target_id": "offline-target", "target_stat": "kl_rms", "target_rms": 1.0}
                        predictions = [{"schema": "bsbench-rms-kl-transfer-v1", "target_id": target["target_id"], "case": {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)}, "method": method, "model": config.get("model_id", "offline"), "target_stat": "kl_rms", "target_rms": 1.0, "bracket": (0.01, 2.0), "predicted_coefficient": 0.3, "search_history": []} for case in PREDICTION_CASES]
                        plans = [final_dose_plan(prediction) for prediction in predictions]
                        records = {PREDICTION_CASES[0].case_id: load_evaluation_records()} | load_transfer_records(); by_case = {plan["case"]["case_id"]: plan for plan in plans}
                        plan = [{"case_id": case.case_id, "target_id": target["target_id"], "coefficient": coefficient, "prompt_id": record.prompt_id, "prompt": record.prompt, "prompt_sha256": record.content_sha256} for case in PREDICTION_CASES for record in records[case.case_id] for coefficient in by_case[case.case_id]["coefficients"]]
                        return {"actual_usd": 0.0, "target": target, "transfer_predictions": predictions, "final_dose_plans": plans, "executable_generation_plan": plan, "baseline_answers": {item["prompt_id"]: "Fake baseline." for item in plan}, "answers": ["Fake final answer." for _ in plan], "health_records": [{"case_id": item["case_id"], "prompt_id": item["prompt_id"], "coefficient": item["coefficient"], "reasons": []} for item in plan], "plan_sha256": content_key({"plan": plan})}
                    raise AssertionError(stage)
            class FakeJudge:
                endpoint = "offline-fake-judge"

                def complete(self, requests):
                    responses = []
                    for request in requests:
                        if request["schema"] == "bsbench-persona-validation-request-v1":
                            responses.append({"intended_behavior_explains": True, "reason": "The paired responses differ on the stated premise."})
                        elif request["blind"]:
                            responses.append({"summary": "offline difference", "changes": []})
                        else:
                            seed = int(hashlib.sha256(request["comparison_id"].encode()).hexdigest()[:8], 16)
                            effect = .15 + (seed % 70) / 100
                            off_target = .02 + ((seed // 70) % 20) / 100
                            if request["order"] == "AB":
                                responses.append({"on_axis_A": 0.0, "on_axis_B": effect, "off_axis_A": 0.0, "off_axis_B": off_target})
                            else:
                                responses.append({"on_axis_A": effect, "on_axis_B": 0.0, "off_axis_A": off_target, "off_axis_B": 0.0})
                    return responses

            backend = FakeBackend()
            ledger = args.ledger or args.out / "costs.jsonl"
            methods = (args.method,) if args.method else METHODS
            common = dict(model={"id": args.model, "judge_model": "offline-fake-judge"}, rows=rows, backend=backend, prompt_spec={"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8}, judge=FakeJudge(), methods=methods)
            first = run_full_sweep(args.out, ledger, **common)
            calls_after_first = list(backend.calls)
            second = run_full_sweep(args.out, ledger, **common)
            result = {"mode": "offline-fake-full-sweep", "not_experimental_results": True,
                      "first_backend_calls": calls_after_first, "immediate_rerun_backend_calls": backend.calls[len(calls_after_first):],
                      "first": first, "second": second}
        elif args.backend == "real":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("real backend requires OPENROUTER_API_KEY before remote callbacks are constructed")
            ledger = args.ledger or args.out / "costs.jsonl"
            manifest = dry_manifest(args.out, args.model, ledger=ledger, cache_aware=True)
            budget = manifest["cost_estimate"]
            from run_bsbench_modal import app, remote_stage_call
            with app.run(), _openrouter_read_timeout(args.openrouter_read_timeout):
                modal_adapter, judge_adapter = real_adapters(
                    modal_stage_call=remote_stage_call(args.model, explicit_run=args.run, budget_preflight=budget),
                    judge_request_call=audited_openrouter_request_callback(endpoint=args.judge_endpoint, api_key=api_key, evidence_root=args.out / "provider-evidence"),
                    judge_endpoint=args.judge_endpoint,
                    explicit_run=args.run,
                    budget_preflight=budget,
                    root=args.out,
                    ledger=ledger,
                )
                rows = read_dev_cohort()
                result = run_full_sweep(
                    args.out,
                    ledger,
                    model={"id": args.model, "judge_model": JUDGE_MODEL},
                    rows=rows,
                    backend=modal_adapter,
                    prompt_spec={"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 128},
                    judge=judge_adapter,
                    methods=(args.method,) if args.method else METHODS,
                )
        else:
            summary = json.loads(Path("slop/verification/20260919_phase6-modal-smoke-summary.json").read_text())
            if summary["model"] != args.model:
                raise ValueError("recorded backend model does not match the completed smoke")
            rows = read_dev_cohort()[:1]
            class RecordedBackend:
                def gpu(self, **_kwargs):
                    return summary["generation_records"][args.method or "vjp_cache"] | {"actual_usd": 0.0}
            ledger = args.ledger or args.out / "costs.jsonl"
            smoke_ledger = Path("outputs/bsbench-smoke/costs.jsonl")
            external = committed(smoke_ledger) if ledger != smoke_ledger else 0.0
            if committed(ledger) + external + 0.001 >= 50.0:
                raise RuntimeError("budget includes the unresolved smoke reservation")
            result = run_stages(args.out, ledger, [{"stage": "generation", "model": {"id": args.model}, "data": cohort_identity(rows), "method": args.method or "vjp_cache", "config": {"upper_usd": 0.001, "recorded_modal_app": summary["modal_app"]}, "prompts": [rows[0]["prompt"]]}], RecordedBackend())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
