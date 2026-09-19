"""Adapters that require explicit run selection and below-limit budget preflight.

The callbacks are injected so importing this module never imports Modal or an API SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from .cache import cached, mark_unresolved, reserve, settle
from .sweep import JUDGE_INPUT_USD_PER_MTOKEN, JUDGE_OUTPUT_USD_PER_MTOKEN


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
            response_body = json.loads(response.read())
        content = response_body["choices"][0]["message"]["content"]
        judgment = json.loads(content)
        schema = payload["response_format"]["json_schema"]["schema"]
        required = set(schema["required"])
        if set(judgment) != required:
            raise ValueError("judge response does not match the strict requested JSON schema")
        expected_types = {"string": str, "number": (int, float), "array": list, "boolean": bool}
        for name, definition in schema["properties"].items():
            if not isinstance(judgment[name], expected_types[definition["type"]]):
                raise ValueError("judge response value does not match the strict requested JSON schema")
        if not isinstance(response_body.get("usage"), dict):
            raise ValueError("judge response omitted usage metadata")
        return judgment | {
            "_remote_usage": response_body["usage"],
            "_remote_cost_usd": response_body.get("usage", {}).get("cost"),
        }

    return call


def judge_request_upper_usd(request: dict) -> float:
    input_tokens = request["input_tokens_upper"] if "input_tokens_upper" in request else (2_000 if request["blind"] else 4_000)
    output_tokens = request["output_tokens_upper"] if "output_tokens_upper" in request else 1_200
    return input_tokens / 1_000_000 * JUDGE_INPUT_USD_PER_MTOKEN + output_tokens / 1_000_000 * JUDGE_OUTPUT_USD_PER_MTOKEN


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
            "schema": "bsbench-judge-request-cache-v1",
            "request": request,
            "judge_model": request["payload"]["model"],
            "judge_endpoint": self.endpoint,
            "upper_usd": upper_usd,
        }

        def compute() -> dict:
            reservation = reserve(self._ledger, f"judge-{request['request_key']}", upper_usd, limit_usd=50.0)
            try:
                response = self._request_call(request["payload"])
            except Exception:
                mark_unresolved(self._ledger, reservation, "judge_request_failure")
                raise
            actual_usd = response.get("_remote_cost_usd")
            if not isinstance(actual_usd, (int, float)) or actual_usd < 0:
                mark_unresolved(self._ledger, reservation, "judge_response_missing_cost")
            else:
                settle(self._ledger, reservation, float(actual_usd))
            return {
                "request": request,
                "response": response,
                "usage": response.get("_remote_usage"),
                "reservation": reservation,
                "upper_usd": upper_usd,
            }

        return cached(self._root / "cache", "judge-request", identity, compute)["response"]

    def complete(self, requests: list[dict]) -> list[dict]:
        self._gate.require()
        return [self._complete_one(request) for request in requests]


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
