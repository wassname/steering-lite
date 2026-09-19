"""Adapters that require explicit run selection and below-limit budget preflight.

The callbacks are injected so importing this module never imports Modal or an API SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable
from urllib.request import Request, urlopen


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


class LocalJudgeAdapter:
    """Adapter for local/API judge calls over persisted existing request payloads."""

    def __init__(self, request_call: Callable[[dict], dict], endpoint: str, gate: RunGate):
        self._request_call = request_call
        self.endpoint = endpoint
        self._gate = gate

    def complete(self, requests: list[dict]) -> list[dict]:
        self._gate.require()
        return [self._request_call(request["payload"]) for request in requests]


def real_adapters(
    *,
    modal_stage_call: Callable[..., dict],
    judge_request_call: Callable[[dict], dict],
    judge_endpoint: str,
    explicit_run: bool,
    budget_preflight: dict,
) -> tuple[ModalRunMethodAdapter, LocalJudgeAdapter]:
    """Build the only real-call path; callers must supply both an explicit run and preflight."""
    gate = RunGate(explicit_run=explicit_run, budget_preflight=budget_preflight)
    return (
        ModalRunMethodAdapter(modal_stage_call, gate),
        LocalJudgeAdapter(judge_request_call, judge_endpoint, gate),
    )
