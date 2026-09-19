"""Content-addressed benchmark artifacts and a local spending ledger. — PI/OpenAI"""

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


def content_key(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hash() -> str:
    root = Path(__file__).parents[1]
    return content_key({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*.py"))})


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def cached(root: Path, stage: str, identity: dict, compute) -> dict:
    key = content_key(identity)
    path = root / stage / f"{key}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("identity") != identity or content_key(record["identity"]) != key:
            raise RuntimeError(f"cache identity mismatch at {path}")
        logger.info("cache hit {} {}", stage, key[:12])
        return record["result"]
    logger.info("cache miss {} {}", stage, key[:12])
    result = compute()
    save_json(path, {"identity": identity, "result": result})
    return result


def cached_stage(
    root: Path,
    stage: str,
    *,
    model: dict,
    data: dict,
    method: str,
    config: dict,
    prompts: list[str],
    compute,
    code: str | None = None,
) -> dict:
    """Cache only when every result-relevant local input is named in the key."""
    if not all((stage, model, data, method, config, prompts)):
        raise ValueError("cached benchmark stage requires model, data, method, config and prompts")
    identity = {
        "schema": "bsbench-stage-v1",
        "stage": stage,
        "model": model,
        "data": data,
        "method": method,
        "config": config,
        "prompts_sha256": content_key({"prompts": prompts}),
        "code_sha256": source_hash() if code is None else code,
    }
    return cached(root, stage, identity, compute)


def committed(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    settled = {row["reservation"]: row["actual_usd"] for row in records if row["event"] == "settled"}
    return sum(settled.get(row["id"], row["upper_usd"]) for row in records if row["event"] == "reserved")


def reserve_many(
    ledger: Path,
    reservations: list[tuple[str, float]],
    limit_usd: float = 49.0,
    *,
    strict_limit: bool = False,
) -> list[str]:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        records = [json.loads(line) for line in handle]
        settled = {row["reservation"]: row["actual_usd"] for row in records if row["event"] == "settled"}
        estimated = {row["reservation"] for row in records if row["event"] == "estimated_at_reservation_upper"}
        unresolved_overages = [row for row in records if row["event"] == "overage"]
        unresolved = [row for row in records if row["event"] == "unresolved" and row["reservation"] not in settled and row["reservation"] not in estimated]
        total = sum(settled.get(row["id"], row["upper_usd"]) for row in records if row["event"] == "reserved")
        requested = sum(upper_usd for _, upper_usd in reservations)
        if unresolved_overages:
            raise RuntimeError("budget: unresolved overage requires audit before another reservation")
        if unresolved:
            raise RuntimeError("budget: unresolved remote work requires a receipt or audit before another reservation")
        if (
            not reservations
            or any(upper_usd <= 0 for _, upper_usd in reservations)
            or total + requested > limit_usd
            or (strict_limit and total + requested >= limit_usd)
        ):
            raise RuntimeError(f"budget: ${total:.4f} committed + ${requested:.4f} requested exceeds ${limit_usd:.2f}")
        records_to_write = []
        for kind, upper_usd in reservations:
            record = {"event": "reserved", "kind": kind, "upper_usd": upper_usd, "time": datetime.now(timezone.utc).isoformat()}
            record["id"] = content_key(record)
            records_to_write.append(record)
        handle.write("".join(json.dumps(record) + "\n" for record in records_to_write))
        return [record["id"] for record in records_to_write]


def reserve(ledger: Path, kind: str, upper_usd: float, limit_usd: float = 49.0) -> str:
    return reserve_many(ledger, [(kind, upper_usd)], limit_usd)[0]


def estimate_at_reservation_upper(ledger: Path, reservation: str, receipt: dict) -> None:
    if not receipt:
        raise ValueError("reservation estimate requires validated provider usage")
    with ledger.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        rows = [json.loads(line) for line in handle]
        reserved, = [row for row in rows if row["event"] == "reserved" and row["id"] == reservation]
        if any(row["event"] == "settled" and row["reservation"] == reservation for row in rows):
            raise ValueError("settled work cannot be estimated")
        if not any(row["event"] == "estimated_at_reservation_upper" and row["reservation"] == reservation for row in rows):
            handle.write(json.dumps({"event": "estimated_at_reservation_upper", "reservation": reservation, "estimated_usd": reserved["upper_usd"], "receipt": receipt}) + "\n")


def mark_unresolved(ledger: Path, reservation: str, reason: str) -> None:
    with ledger.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        rows = [json.loads(line) for line in handle]
        if not any(row["event"] == "reserved" and row["id"] == reservation for row in rows):
            raise ValueError("unresolved work has no matching reservation")
        if any(row["event"] == "settled" and row["reservation"] == reservation for row in rows):
            return
        if not any(row["event"] == "unresolved" and row["reservation"] == reservation for row in rows):
            handle.write(json.dumps({"event": "unresolved", "reservation": reservation, "reason": reason}) + "\n")


def settle(ledger: Path, reservation: str, actual_usd: float) -> None:
    if actual_usd < 0:
        raise ValueError("negative cost")
    with ledger.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        rows = [json.loads(line) for line in handle]
        original, = [row for row in rows if row["event"] == "reserved" and row["id"] == reservation]
        assert not any(row["event"] == "settled" and row["reservation"] == reservation for row in rows)
        handle.write(json.dumps({"event": "settled", "reservation": reservation, "actual_usd": actual_usd}) + "\n")
        if actual_usd > original["upper_usd"]:
            handle.write(json.dumps({"event": "overage", "reservation": reservation, "actual_usd": actual_usd, "upper_usd": original["upper_usd"]}) + "\n")
            raise RuntimeError(f"actual cost ${actual_usd} exceeded reservation ${original['upper_usd']}; revise pricing before continuing")


def settle_receipt(ledger: Path, reservation: str, actual_usd: float, receipt: dict) -> None:
    if not receipt:
        raise ValueError("receipt import requires receipt metadata")
    settle(ledger, reservation, actual_usd)
    with ledger.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps({"event": "receipt_imported", "reservation": reservation, "receipt": receipt}) + "\n")
