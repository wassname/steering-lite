"""Check wording-only normal rebuilds preserve every scientific JSON byte. — PI/OpenAI"""
import hashlib
import json
from pathlib import Path

expected = json.loads(Path(__file__).with_name("legend-points-before.json").read_text())
for report, digest in expected.items():
    path = Path("outputs/bsbench/results") / report / "points.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, report
    print(f"CAPTION_DATA_IDENTICAL: {report} points, scores, intervals, selected doses, raw bounds and plot support byte-identical")
