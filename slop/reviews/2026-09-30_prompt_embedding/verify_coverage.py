"""Recheck saved prompt-sweep coverage and exact fresh controls. -- PI/OpenAI"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "outputs/bsbench/Qwen--Qwen3.5-4B-g7c7712c6"
REPORT = Path(__file__).with_name("verification.json")
GAINS = [0, .125, .25, .5, 1, 2, 4, 8, 16]
cohort = [json.loads(line) for line in (ROOT / "data/bsbench/bullshit_bench_v2.jsonl").read_text().splitlines()][::5]
expected = {row["scenario"] for row in cohort}
assert len(expected) == 20
verification = json.loads(REPORT.read_text())
keys = set()
for method in ("prompting_scale", "prompting_engineered_scale"):
    certificate = MODEL / f"walks/{method}_s0_dev.json"
    walk = json.loads(certificate.read_text())
    assert walk["status"] == "COMPLETE" and walk["seed"] == 0 and walk["cohort"] == "dev"
    assert walk["sweep_kind"] == "prompt_embeddings" and not walk["boundary_confirmed"]
    assert [r["coefficient"] for r in walk["rungs"]] == GAINS
    sources, run_ids = [], set()
    for rung in walk["rungs"]:
        for side in ("+C", "-C"):
            path = MODEL / rung[side]["answers"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 20 and {r["scenario"] for r in rows} == expected
            for row in rows:
                key = (method, side, rung["coefficient"], row["scenario"])
                assert key not in keys
                keys.add(key)
                if method == "prompting_engineered_scale":
                    run_ids.add(row["run_id"])
            sources.append({"path": str(path.relative_to(ROOT)), "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    verification[method].update(status=walk["status"], unique_rows=360, certificate_sha256=hashlib.sha256(certificate.read_bytes()).hexdigest(), sources=sources)
    if method == "prompting_engineered_scale":
        assert len(run_ids) == 1
        verification[method]["run_ids"] = sorted(run_ids)
        for side in ("+C", "-C"):
            rows = json.loads((MODEL / walk["identity"][side]["diagnostic"]).read_text())
            assert len(rows) == 20 and {r["scenario"] for r in rows} == expected
            assert all(r["fresh"] == r["scaled_C1"] for r in rows)
            assert sum(r["fresh"] != r["historical"] for r in rows) == walk["identity"][side]["historical_mismatches"]
assert len(keys) == 720
verification["unique_rows"] = len(keys)
REPORT.write_text(json.dumps(verification, indent=2) + "\n")
print("COVERAGE_PASS: 720 unique rows; complete 20-question x 9-gain x 2-sign x 2-method grid; fresh engineered identity 40/40; one engineered process")
