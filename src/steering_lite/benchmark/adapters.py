"""Explicitly gated adapters for the planned Modal GPU and local judge APIs.

The callbacks are injected so importing this module never imports Modal or an API SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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
