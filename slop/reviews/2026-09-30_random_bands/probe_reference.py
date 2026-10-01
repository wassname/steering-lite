"""Compare cached 11/16-seed full random references, using production scoring; no API calls. PI/OpenAI."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bsbench"))
import results

root = results.default_model_dir()
certificates = [json.loads(p.read_text()) for p in sorted((root / "walks").glob("random_s*_full.json"))]
certificates = [c for c in certificates if c["status"] == "COMPLETE" and c["seed"] < 16]
assert {c["seed"] for c in certificates} == set(range(16))
with patch("results.walk_certificates", return_value=certificates):
    points = results.build_points(root, "full", set())
references = {}
for count in (11, 16):
    zones = results.random_zones([p for p in points if p["seed"] < count])
    references[count] = {z["percentile"]: {C: {"bounds": b, "seeds": n}
                         for C, b, n in zip(z["doses"][1:], z["bounds"][1:], z["seed_counts"][1:], strict=True)} for z in zones}
    print(f"FULL_REFERENCE seeds={count} questions=100 supported_doses={len(zones[0]['doses']) - 1}")
for C in sorted(references[11][90].keys() & references[16][90].keys()):
    old, new = references[11][90][C], references[16][90][C]
    print(f"C={C:.4g} coherent_seeds={old['seeds']}->{new['seeds']} p90_left={old['bounds'][2]:+.3f}->{new['bounds'][2]:+.3f} p90_right={old['bounds'][3]:+.3f}->{new['bounds'][3]:+.3f} median_damage={old['bounds'][1]:.3f}->{new['bounds'][1]:.3f}")
Path(__file__).with_name("reference-probe.json").write_text(json.dumps({"cohort": "full", "questions": 100, "source": str(root), "seed_ranges": ["0-10", "0-15"], "references": references}, indent=2) + "\n")
