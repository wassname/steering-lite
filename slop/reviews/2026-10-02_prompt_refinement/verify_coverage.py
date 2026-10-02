"""Audit preserved bytes and exact dev coverage through production loaders. — PI/OpenAI"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/bsbench"))
from data import COHORTS, demo_rows, load_cohort, walk_certificates
from walk import BASE_PROMPT_GAINS, PROMPT_GAINS

root = Path("outputs/bsbench/Qwen--Qwen3.5-4B-g7c7712c6")
hashes = json.loads(Path(__file__).with_name("historical-hashes.json").read_text())
for path, expected in hashes.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
print(f"HASH_PASS: {len(hashes)} historical answer/full-certificate files unchanged")
expected_scenarios = set(list(load_cohort())[COHORTS["dev"]])
certificates = walk_certificates(root, "dev")
random = [c for c in certificates if c["method"] == "random"]
assert {c["seed"] for c in random} == set(range(32))
prompt = next(c for c in certificates if c["method"] == "prompting_scale")
engineered = next(c for c in certificates if c["method"] == "prompting_engineered_scale")
assert prompt["prompt_gains"] == list(PROMPT_GAINS["prompting_scale"])
assert engineered["prompt_gains"] == list(BASE_PROMPT_GAINS)
for c in [prompt, engineered, *random]:
    assert c["status"] == "COMPLETE"
    rows = demo_rows(root, c)
    keys = [(r["C"], r["side"], r["vignette"]) for r in rows]
    assert len(keys) == len(set(keys)) == len(c["rungs"]) * 2 * 20
    for dose in c["rungs"]:
        for side in ("+C", "-C"):
            path = root / dose[side]["answers"]
            records = [json.loads(line) for line in path.open()]
            ids = [r["scenario"] for r in records]
            assert len(ids) == len(set(ids)), path
            assert expected_scenarios <= set(ids), path
            if c is prompt and dose["coefficient"] not in BASE_PROMPT_GAINS:
                assert set(ids) == expected_scenarios
                assert dose[side]["answer_runs"] == [c["run_id"]]
                assert {r["run_id"] for r in records} == {c["run_id"]}
    print(f"COVERAGE_PASS: {c['method']} seed={c['seed']} rungs={len(c['rungs'])} dev_rows={len(rows)}")
assert len(prompt["rungs"]) == 31
assert all(p["exact"] and p["answers"] == 20 and p["historical_mismatches"] == 0 for p in prompt["identity"].values())
print("IDENTITY_PASS: fresh ordinary vs cached scaled gain-one 40/40; historical mismatches 0; same-process fresh identity separately checked on first 3 prompts; original nine gains preserved")
new = [prompt, *(c for c in random if c["seed"] >= 11)]
seconds = sum(c["timing"]["total_s"] for c in new)
print(f"COST_PROXY: worker-total seconds={seconds:.3f}; L40S GPU-only list-price proxy=${seconds * .000542:.4f}; not invoice; excludes wrapper/CPU/memory")
