"""Resume only the saved mean_diff final judge requests. — PI/gpt-6-sol"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections import Counter
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path

from steering_lite.benchmark import adapters, production
from steering_lite.benchmark.adapters import judge_cache_identity, judge_request_cached, judge_request_upper_usd, real_adapters
from steering_lite.benchmark.cache import committed, content_key, require_resolved_ledger, source_hash
from steering_lite.benchmark.generation import read_dev_cohort
from steering_lite.benchmark.sweep import JUDGE_MODEL, MODEL_ID, PHASE6_SMOKE_LEDGER, load_judge_pricing

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_bsbench_sweep as runner

ROOT = Path("outputs/bsbench-v2-mean-diff-cap384")
LEDGER = Path("outputs/bsbench-v2/costs.jsonl")
PRICING = Path("slop/verification/20260922_v4-provider-endpoint-metadata.json")
CACHE_AUDIT = Path("slop/verification/20260923_mean_diff_final_failure_cache_audit.json")
LEDGER_AUDIT = Path("slop/verification/20260923_mean_diff_final_failure_ledger_audit.json")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class DryJudgeBoundary(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-probe", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    cache_audit = json.loads(CACHE_AUDIT.read_text())
    ledger_audit = json.loads(LEDGER_AUDIT.read_text())
    assert source_hash() == cache_audit["source_sha256"] == "eff29d45f8bdef895a652196eb284d47a39966447ec77471a1b8876ad5c96497"
    assert digest(Path("scripts/run_bsbench_sweep.py")) == "7f82e82a205cfe0d7fa1b2a6d04d9df641e4a19e6228a847be3048beec5a056f"
    assert digest(Path("scripts/run_bsbench_modal.py")) == "1c9a251cedf9e5a004a844fea32f7cb34d86eb9cac22204b7c9885549372a296"
    assert digest(PRICING) == cache_audit["judge_pricing_sha256"]
    assert digest(LEDGER) == ledger_audit["ledger_sha256"]
    assert digest(Path(cache_audit["final_generation_path"])) == cache_audit["final_generation_sha256"]
    require_resolved_ledger(LEDGER)
    require_resolved_ledger(PHASE6_SMOKE_LEDGER)
    load_judge_pricing(PRICING)
    states = cache_audit["requests"]
    assert len(states) == 720 and len({s["key"] for s in states}) == 720
    missing = {s["evidence_id"]: s for s in states if not s["cached"]}
    assert len(missing) == 333 and sum(s["cached"] for s in states) == 387
    upper = sum((Decimal(str(s["attempt_upper_usd"])) for s in missing.values()), Decimal("0"))
    assert upper == Decimal("0.374232") and 3 * upper == Decimal(ledger_audit["missing_three_attempt_upper_usd"])
    existing = Decimal(str(committed(LEDGER)))
    external = Decimal(str(committed(PHASE6_SMOKE_LEDGER)))
    hold = Decimal(ledger_audit["ancillary_hold_usd"])
    assert abs(existing - Decimal(ledger_audit["committed_ledger_usd"])) < Decimal("0.000000001")
    assert external == Decimal("2") and hold == Decimal("1")
    total = existing + external + hold + 3 * upper
    assert total < Decimal("50")
    budget = {"total_upper_usd": float(total), "limit_usd": 50.0, "external_committed_usd": float(external + hold)}
    ledger_before = digest(LEDGER)
    files_before = {str(path.relative_to(ROOT)): digest(path) for path in ROOT.rglob("*") if path.is_file()}
    lock = threading.Lock()
    attempts = Counter()
    reserved_upper = Decimal("0")
    validated = False
    denied = 0
    original_modal_reserve = production.reserve
    original_judge_reserve = adapters.reserve

    def forbid_modal_reservation(*unused, **kwargs):
        raise RuntimeError("GPU reservation denied before ledger mutation")

    def guarded_judge_reservation(ledger: Path, kind: str, amount: float, **kwargs):
        nonlocal reserved_upper, denied
        assert ledger == LEDGER
        match = re.fullmatch(r"judge-([0-9a-f]{64})-attempt-([123])", kind)
        if match is None or match[1] not in missing:
            raise RuntimeError(f"unapproved judge reservation: {kind}")
        state = missing[match[1]]
        with lock:
            if Decimal(str(amount)) != Decimal(str(state["attempt_upper_usd"])) or int(match[2]) != attempts[match[1]] + 1:
                raise RuntimeError("judge attempt differs from pinned request upper or retry sequence")
            if reserved_upper + Decimal(str(amount)) > 3 * upper:
                raise RuntimeError("affected-requests upper bound exhausted")
            attempts[match[1]] += 1
            reserved_upper += Decimal(str(amount))
            if args.dry_probe:
                denied += 1
                raise DryJudgeBoundary("first uncached judge request reached before reservation")
        return original_judge_reserve(ledger, kind, amount, **kwargs)

    def forbid_gpu_callback(**kwargs):
        raise RuntimeError("GPU dispatch denied")

    def forbid_judge_callback(payload: dict):
        raise RuntimeError("provider dispatch denied during offline probe")

    callback = forbid_judge_callback if args.dry_probe else runner.audited_openrouter_request_callback(
        endpoint=ENDPOINT, api_key=os.environ["OPENROUTER_API_KEY"], evidence_root=ROOT / "provider-evidence"
    )
    modal, judge = real_adapters(
        modal_stage_call=forbid_gpu_callback,
        judge_request_call=callback,
        judge_endpoint=ENDPOINT,
        explicit_run=True,
        budget_preflight=budget,
        root=ROOT,
        ledger=LEDGER,
    )
    original_complete = judge.complete

    def checked_complete(requests: list[dict]) -> list[dict]:
        nonlocal validated
        assert not validated and len(requests) == len(states)
        for request, state in zip(requests, states, strict=True):
            identity = judge_cache_identity(request, ENDPOINT)
            if content_key(identity) != state["key"] or identity["evidence_id"] != state["evidence_id"]:
                raise RuntimeError("final request differs from saved exact-identity audit")
            if judge_request_upper_usd(request) != state["attempt_upper_usd"]:
                raise RuntimeError("pinned request upper changed")
            if judge_request_cached(ROOT, request, ENDPOINT) != state["cached"]:
                raise RuntimeError("cached/missing membership changed before paid request")
        validated = True
        return original_complete(requests)

    judge.complete = checked_complete
    production.reserve = forbid_modal_reservation
    adapters.reserve = guarded_judge_reservation
    try:
        with nullcontext() if args.dry_probe else runner._openrouter_read_timeout(180.0):
            result = runner.run_full_sweep(
                ROOT, LEDGER, model={"id": MODEL_ID, "judge_model": JUDGE_MODEL}, rows=read_dev_cohort(),
                backend=modal, prompt_spec={"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 128},
                judge=judge, methods=("mean_diff",),
            )
    except DryJudgeBoundary:
        if not args.dry_probe or not validated or denied == 0 or digest(LEDGER) != ledger_before:
            raise
        if {str(path.relative_to(ROOT)): digest(path) for path in ROOT.rglob("*") if path.is_file()} != files_before:
            raise RuntimeError("dry probe modified production outputs or provider evidence")
        print(json.dumps({"dry_probe": "denied_before_first_judge_reservation", "validated_requests": len(states), "cached": 387, "uncached": len(missing), "denied_reservations": denied, "ledger_unchanged": True, "outputs_unchanged": True, "budget_including_ancillary_usd": str(total)}, sort_keys=True))
    else:
        if args.dry_probe or not validated or digest(LEDGER) == ledger_before or not result["conditions"]["mean_diff"]["final_judgments"]:
            raise RuntimeError("bounded resume did not complete as expected")
        print(json.dumps({"run": "complete", "summary_path": result["summary_path"], "new_judge_reservations": sum(attempts.values()), "reserved_upper_usd": str(reserved_upper)}, sort_keys=True))
    finally:
        production.reserve = original_modal_reserve
        adapters.reserve = original_judge_reserve


if __name__ == "__main__":
    main()
