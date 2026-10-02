"""Verify Jev-only filtering and unchanged measured answers/ratings. — PI/OpenAI"""
import hashlib
import json
from pathlib import Path

hashes = json.loads(Path("slop/reviews/2026-10-02_prompt_refinement/historical-hashes.json").read_text())
for name, digest in hashes.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
print(f"HISTORICAL_BYTES_PASS: {len(hashes)} answers and full certificates unchanged")
for name in ["dev", "prompt-dev", "full", "27b-full", "olmo-full"]:
    before = json.loads(Path(f".local/jev-only-before/{name}.json").read_text())
    after = json.loads(Path(f"outputs/bsbench/results/{name}/points.json").read_text())
    assert after["admissibility"] == "jev_mean_damage"
    identity = lambda p: (p["method"], p["seed"], p["side"], p["C"])
    old = {identity(p): p for p in before["points"]}
    new = {identity(p): p for p in after["points"]}
    assert old.keys() == new.keys()
    admitted = []
    for key, p in new.items():
        b = old[key]
        assert p["admissible"] == (p["steered_damage"] <= after["max_damage"]), key
        assert {k:v for k,v in b.items() if k not in ("admissible", "questions")} == {k:v for k,v in p.items() if k not in ("admissible", "questions")}
        for qb, qa in zip(b["questions"], p["questions"], strict=True):
            assert {k:v for k,v in qb.items() if k != "blind"} == {k:v for k,v in qa.items() if k != "blind"}
            assert qb["blind"] is None or qb["blind"] == qa["blind"]
        if p["admissible"] and not b["admissible"]:
            admitted.append({"method":p["method"],"seed":p["seed"],"side":p["side"],"C":p["C"],"damage":p["steered_damage"],"mechanical":p["breakdown_reasons"],"post_boundary":p["post_boundary"]})
    print("JEV_ONLY_FILTER_PASS", name, "newly_admitted", len(admitted), admitted)
    previous = {r["method"]: r for r in before["summary"]}
    for row in after["summary"]:
        b = previous[row["method"]]
        if any(b[k] != row[k] for k in ("score", "best", "N", "rejected")):
            print("SUMMARY_CHANGED", name, row["method"], {k:{"before":b[k],"after":row[k]} for k in ("score","best","N","rejected") if b[k]!=row[k]})
    print("RAW_MEASUREMENTS_PASS", name, "all answers, premise/damage ratings, mechanical diagnostics and prior blind ratings unchanged")
