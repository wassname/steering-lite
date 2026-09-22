"""Adapters that require explicit run selection and below-limit budget preflight.

The callbacks are injected so importing this module never imports Modal or an API SDK.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from http.client import RemoteDisconnected
import hashlib
import json
from pathlib import Path
from socket import timeout as SocketTimeout
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cache import cached, estimate_at_reservation_upper, reserve_many, settle
from .judge import RETRY_NUDGE
from .sweep import JUDGE_INPUT_USD_PER_MTOKEN, JUDGE_OUTPUT_USD_PER_MTOKEN


class OpenRouterResponseParseError(ValueError):
    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True)
class RunGate:
    """Permit a real remote call only after the CLI has selected a bounded run."""

    explicit_run: bool
    budget_preflight: dict

    def require(self) -> None:
        if not self.explicit_run:
            raise RuntimeError("real backend requires an explicit --run selection")
        if self.budget_preflight["total_upper_usd"] >= self.budget_preflight["limit_usd"]:
            raise RuntimeError("real backend requires a budget preflight below its limit")


class ModalRunMethodAdapter:
    """Adapter for a Modal stage callback that implements the existing run_method path."""

    remote_vector_binding = True
    paid_execution_enabled = True

    def __init__(self, stage_call: Callable[..., dict], gate: RunGate):
        self._stage_call = stage_call
        self._gate = gate

    def gpu(self, *, stage: str, method: str, config: dict, prompts: list[str]) -> dict:
        self._gate.require()
        return self._stage_call(stage=stage, method=method, config=config, prompts=prompts)


def openrouter_request_callback(*, endpoint: str, api_key: str) -> Callable[[dict], dict]:
    """Return the one-shot OpenRouter callback for the already-persisted request payload."""
    if not endpoint.startswith("https://"):
        raise ValueError("judge endpoint must use HTTPS")
    if not api_key:
        raise ValueError("OpenRouter API key is required")

    def call(payload: dict) -> dict:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        request = Request(endpoint, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=90) as response:
            raw_response = response.read()
        try:
            response_body = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise OpenRouterResponseParseError(
                "OpenRouter response body is not JSON",
                {
                    "response_sha256": hashlib.sha256(raw_response).hexdigest(),
                    "response_bytes": len(raw_response),
                    "response_excerpt": raw_response.decode(errors="replace")[:10_000],
                    "json_error": str(error),
                },
            ) from error
        try:
            content = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenRouterResponseParseError(
                "OpenRouter response lacks assistant content",
                {
                    "response_sha256": hashlib.sha256(raw_response).hexdigest(),
                    "response_bytes": len(raw_response),
                    "response": response_body,
                    "structural_error": str(error),
                },
            ) from error
        try:
            judgment = json.loads(content)
        except json.JSONDecodeError as error:
            raise OpenRouterResponseParseError(
                "OpenRouter assistant content is not strict JSON",
                {
                    "response_sha256": hashlib.sha256(raw_response).hexdigest(),
                    "assistant_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "assistant_content_bytes": len(content.encode()),
                    "assistant_content": content[:10_000],
                    "json_error": str(error),
                },
            ) from error
        schema = payload["response_format"]["json_schema"]["schema"]
        required = set(schema["required"])
        if set(judgment) != required:
            raise OpenRouterResponseParseError("judge response does not match the strict requested JSON schema", {"judgment": judgment})
        expected_types = {"string": str, "number": (int, float), "array": list, "boolean": bool}
        for name, definition in schema["properties"].items():
            if not isinstance(judgment[name], expected_types[definition["type"]]):
                raise OpenRouterResponseParseError("judge response value does not match the strict requested JSON schema", {"field": name, "judgment": judgment})
        if not isinstance(response_body.get("usage"), dict):
            raise OpenRouterResponseParseError("judge response omitted usage metadata", {"response": response_body})
        return judgment | {
            "_remote_usage": response_body["usage"],
            "_remote_cost_usd": response_body.get("usage", {}).get("cost"),
        }

    return call


TRANSIENT_CODES = {408, 429, 500, 502, 503, 504, 524, 529}
JUDGE_ATTEMPTS = 3
JUDGE_CONCURRENCY = 6
JUDGE_RETRY_POLICY = {
    "attempts": JUDGE_ATTEMPTS,
    "aware_format_rescue_attempts": [2, 3],
    "aware_retry_nudge": RETRY_NUDGE,
    "blind_and_persona_payloads_unchanged": True,
}


def _attempt_payload(request: dict, attempt: int) -> dict:
    payload = deepcopy(request["payload"])
    if request.get("blind") is False and attempt > 1:
        payload["messages"][0]["content"] += RETRY_NUDGE
    return payload


def judge_request_upper_usd(request: dict) -> float:
    if JUDGE_INPUT_USD_PER_MTOKEN is None or JUDGE_OUTPUT_USD_PER_MTOKEN is None:
        raise RuntimeError("judge pricing is unsourced for deepseek/deepseek-v4-flash-0731; paid dispatch is disabled")
    input_tokens = request["input_tokens_upper"] if "input_tokens_upper" in request else (2_000 if request["blind"] else 4_000)
    output_tokens = request["output_tokens_upper"] if "output_tokens_upper" in request else 1_024
    return input_tokens / 1_000_000 * JUDGE_INPUT_USD_PER_MTOKEN + output_tokens / 1_000_000 * JUDGE_OUTPUT_USD_PER_MTOKEN


def _retryable(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code in TRANSIENT_CODES
    return isinstance(error, (OpenRouterResponseParseError, TimeoutError, SocketTimeout, URLError, RemoteDisconnected, ConnectionError))


def _attempt_receipt(request: dict, attempt: int, error: Exception) -> dict:
    receipt = {
        "schema": "bsbench-judge-attempt-failure-v1",
        "request_key": request["request_key"],
        "attempt": attempt,
        "exception_type": type(error).__name__,
    }
    if isinstance(error, HTTPError):
        receipt["status"] = error.code
    if isinstance(error, OpenRouterResponseParseError):
        receipt["parse_evidence"] = error.evidence
    if hasattr(error, "evidence_path"):
        receipt["provider_evidence"] = error.evidence_path
    return receipt


class LocalJudgeAdapter:
    """Adapter for local/API judge calls over persisted existing request payloads."""

    def __init__(self, request_call: Callable[[dict], dict], endpoint: str, gate: RunGate, *, root: Path, ledger: Path):
        self._request_call = request_call
        self.endpoint = endpoint
        self._gate = gate
        self._root = root
        self._ledger = ledger

    def _complete_one(self, request: dict) -> dict:
        upper_usd = judge_request_upper_usd(request)
        identity = {
            "schema": "bsbench-judge-request-cache-v2",
            "request": request,
            "judge_model": request["payload"]["model"],
            "judge_endpoint": self.endpoint,
            "upper_usd": upper_usd,
            "retry_policy": JUDGE_RETRY_POLICY,
        }

        def compute() -> dict:
            reservations = reserve_many(
                self._ledger,
                [(f"judge-{request['request_key']}-attempt-{attempt}", upper_usd) for attempt in range(1, JUDGE_ATTEMPTS + 1)],
                limit_usd=50.0,
            )
            failures = []
            for attempt, reservation in enumerate(reservations, 1):
                try:
                    response = self._request_call(_attempt_payload(request, attempt))
                except Exception as error:
                    receipt = _attempt_receipt(request, attempt, error)
                    estimate_at_reservation_upper(self._ledger, reservation, receipt)
                    failures.append(receipt)
                    if attempt < JUDGE_ATTEMPTS and _retryable(error):
                        continue
                    for unused in reservations[attempt:]:
                        settle(self._ledger, unused, 0.0)
                    raise
                actual_usd = response.get("_remote_cost_usd")
                if not isinstance(actual_usd, (int, float)) or actual_usd < 0:
                    estimate_at_reservation_upper(self._ledger, reservation, {"schema": "bsbench-judge-attempt-missing-cost-v1", "request_key": request["request_key"], "attempt": attempt, "usage": response.get("_remote_usage")})
                else:
                    settle(self._ledger, reservation, float(actual_usd))
                for unused in reservations[attempt:]:
                    settle(self._ledger, unused, 0.0)
                return {
                    "request": request,
                    "response": response,
                    "usage": response.get("_remote_usage"),
                    "reservation": reservation,
                    "attempt_reservations": reservations,
                    "failed_attempts": failures,
                    "upper_usd": upper_usd,
                }
            raise RuntimeError("judge request exhausted attempts")

        return cached(self._root / "cache", "judge-request", identity, compute)["response"]

    def complete(self, requests: list[dict]) -> list[dict]:
        self._gate.require()
        with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as executor:
            return list(executor.map(self._complete_one, requests))


def real_adapters(
    *,
    modal_stage_call: Callable[..., dict],
    judge_request_call: Callable[[dict], dict],
    judge_endpoint: str,
    explicit_run: bool,
    budget_preflight: dict,
    root: Path,
    ledger: Path,
) -> tuple[ModalRunMethodAdapter, LocalJudgeAdapter]:
    """Build the only real-call path; callers must supply both an explicit run and preflight."""
    gate = RunGate(explicit_run=explicit_run, budget_preflight=budget_preflight)
    return (
        ModalRunMethodAdapter(modal_stage_call, gate),
        LocalJudgeAdapter(judge_request_call, judge_endpoint, gate, root=root, ledger=ledger),
    )
