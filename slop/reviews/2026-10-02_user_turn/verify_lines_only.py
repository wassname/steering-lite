"""Plot-line change only: scientific content of points.json unchanged, except audit ratings newly
attached from the Jev cache (None -> rating; same answer text rated by compare.py). PI/OpenAI 2026-10-03."""
import json
import sys

out = sys.argv[1]
a = json.load(open(f".local/lines-before/{out}.json"))
b = json.load(open(f"outputs/bsbench/results/{out}/points.json"))
assert a["summary"] == b["summary"] and a["zones"] == b["zones"] and len(a["points"]) == len(b["points"])
attached = 0
for x, y in zip(a["points"], b["points"]):
    assert {k: v for k, v in x.items() if k != "questions"} == {k: v for k, v in y.items() if k != "questions"}
    for q, r in zip(x["questions"], y["questions"], strict=True):
        assert {k: v for k, v in q.items() if k != "audit"} == {k: v for k, v in r.items() if k != "audit"}
        if q.get("audit") != r.get("audit"):
            assert q.get("audit") is None, "an existing audit rating changed"
            attached += 1
print(f"LINES_ONLY_PASS {out}: scores, CIs, selections, points, ratings and random zones identical; {attached} cached audit ratings newly attached")
