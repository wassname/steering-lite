"""Verify renamed cached configs, CLI IDs and saved statistics. PI/OpenAI."""

import json
import sys
from pathlib import Path

from migrate_artifacts import NAMES, metadata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bsbench"))
import steering_lite as sl
import walk

for name in NAMES.values():
    assert name in sl.REGISTRY
    assert walk.parse_args([name]).method == name
assert not set(NAMES) & sl.REGISTRY.keys()

manifest = json.loads((ROOT / "slop/reviews/method-naming/local-migration.json").read_text())
vectors = [r for r in manifest if r["new"].endswith(".safetensors")]
for record in vectors:
    path = ROOT / "outputs/bsbench" / record["new"]
    with path.open("rb") as file:
        size = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(size))
    cfg = sl.SteeringConfig.from_dict(json.loads(header["__metadata__"]["cfg"]))
    assert cfg.method in NAMES.values(), path
print(f"All 7 new CLI IDs registered, old IDs absent; {len(vectors)} renamed vector configs deserialize")

for before in (ROOT / ".local/method-naming").glob("*-before.json"):
    cohort = before.name.removesuffix("-before.json")
    expected = metadata(json.loads(before.read_text()))
    actual = json.loads((ROOT / "outputs/bsbench/results" / cohort / "points.json").read_text())
    for key, value in expected.items():
        assert actual[key] == value, (cohort, key)
    print(f"{cohort}: summary, curves, method selection, colors and blind ratings exactly match pre-rename snapshot")
