"""Content-addressed benchmark artifacts and a local spending ledger. — PI/OpenAI"""

import fcntl
import hashlib
import json
import math
from numbers import Real
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
    compatible_code_sha256s: tuple[str, ...] = (),
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
    if compatible_code_sha256s:
        semantic_identity = {key: value for key, value in identity.items() if key != "code_sha256"}
        matches = []
        for path in (root / stage).glob("*.json"):
            record = json.loads(path.read_text())
            cached_identity = record.get("identity")
            if not isinstance(cached_identity, dict) or content_key(cached_identity) != path.stem:
                raise RuntimeError(f"cache identity mismatch at {path}")
            if cached_identity.get("code_sha256") not in compatible_code_sha256s:
                continue
            if {key: value for key, value in cached_identity.items() if key != "code_sha256"} == semantic_identity:
                matches.append((path, record))
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous compatible cache records for {stage} {method}")
        if matches:
            path, record = matches[0]
            logger.info("cache reuse compatible {} {}", stage, path.stem[:12])
            return record["result"]
    return cached(root, stage, identity, compute)


def valid_cost(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_cost_records(records: list[dict]) -> None:
    for row in records:
        for field in ("upper_usd", "actual_usd", "estimated_usd"):
            if field in row and (not valid_cost(row[field]) or (row["event"] == "reserved" and row[field] == 0)):
                raise ValueError(f"invalid ledger cost: {field}")


def require_resolved_ledger(ledger: Path) -> None:
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    validate_cost_records(rows)
    resolved = {row["reservation"] for row in rows if row["event"] in {"settled", "estimated_at_reservation_upper"}}
    if any(row["event"] == "overage" or (row["event"] == "reserved" and row["id"] not in resolved) for row in rows):
        raise RuntimeError("ledger has unresolved work or overage")


def committed(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    validate_cost_records(records)
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
        validate_cost_records(records)
        if not valid_cost(limit_usd) or not reservations or any(not valid_cost(cost) or cost == 0 for _, cost in reservations):
            raise ValueError("reservation upper and limit must be finite positive costs")
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
    return reserve_many(ledger, [(kind, upper_usd)], limit_usd, strict_limit=True)[0]


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
    if not valid_cost(actual_usd):
        raise ValueError("actual cost must be finite nonnegative and non-boolean")
    with ledger.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        rows = [json.loads(line) for line in handle]
        validate_cost_records(rows)
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
