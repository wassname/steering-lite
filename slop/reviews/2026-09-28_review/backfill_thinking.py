"""Add "thinking" to vector sidecar jsons written before walk.py recorded it (review round 1, item 1).

Every extraction before commit ba717fd (walk.py --no-think) used make_persona_pairs(thinking=True);
the one no-think extraction is vjp_delta-nothink_s0 (OLMo). Key order follows walk.extract_vector.
Run: python3 backfill_thinking.py <vectors dir> [<vectors dir> ...]   (PI/Claude, 2026-09-28)
"""
import json
import sys
from pathlib import Path

for directory in sys.argv[1:]:
    paths = sorted(Path(directory).glob("*.json"))
    assert paths, f"no vector jsons in {directory}"
    for path in paths:
        saved = json.loads(path.read_text())
        assert "thinking" not in saved, f"{path} already has thinking={saved['thinking']}"
        thinking = path.name != "vjp_delta-nothink_s0.json"
        head = {key: saved.pop(key) for key in ("method", "seed", "layers", "n_pairs")}
        path.write_text(json.dumps({**head, "thinking": thinking, **saved}, indent=2) + "\n")
        print(f"{path} thinking={thinking}")
