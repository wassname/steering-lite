"""Run metadata + audit log: timestamp, git, argv, hostname for every output JSON.

Lets a future analyst grep `outputs/tinymfv_sweep/runs.jsonl` and reconstruct
"which command produced this number, on which commit, when". Append-only;
each line = one method completion. Same `run_id` ties methods within a sweep.
"""
from __future__ import annotations

import datetime
import json
import socket
import subprocess
import sys
import uuid
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def make_metadata(args, **extra) -> dict:
    """Per-run metadata. Stable across all methods within a single sweep call.

    Args is an argparse.Namespace; Path values get stringified for JSON.
    Extra kwargs land in the returned dict as-is.
    """
    args_dict = vars(args).copy() if hasattr(args, "__dict__") else dict(args)
    for k, v in list(args_dict.items()):
        if isinstance(v, Path):
            args_dict[k] = str(v)
    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "args": args_dict,
        **extra,
    }


def append_run(out_dir: Path, record: dict) -> None:
    """Append a single one-line JSONL record to runs.jsonl. Append-only audit log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "runs.jsonl"
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")
