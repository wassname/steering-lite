"""Persona-pair extraction data on nonsense questions (BullshitBench v1), not on the generic suffix file.

Why: the default pairs (steering_lite/data/branching_suffixes_filt.json: "Tell me a story.", job descriptions, ...)
contain no nonsense premise, so "explains why, when a question's premise doesn't make sense" has nothing to act on
there. Here the shared user message is a BullshitBench v1 question (55, none in the v2 eval set) with the eval's
answer suffix, and the shared assistant start is Qwen3.5-4B's own unsteered answer cut after 2, 4, 8 and 16 words
(truncated-output pairs, as in repeng). 220 entries. PI/OpenAI 2026-10-04.

    uv run --extra benchmark python scripts/bsbench/build_pairs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / ".local/bsb_v1_questions.json"  # petergpt/bullshit-benchmark @ 6f6e28b4 questions.json
BARE = ROOT / "outputs/bsbench/Qwen--Qwen3.5-4B-g1f092bc2/answers/bare/bare.jsonl"  # its v1 rows: bullshit_v1_<id>
OUT = ROOT / "data/bsbench/persona_pairs_bsbench_v1.json"
SUFFIX = " Answer in 2 short sentences."  # data.GEN["suffix"]

bare = {r["scenario"]: r["text"] for r in map(json.loads, BARE.open())}
entries = []
for technique in json.loads(QUESTIONS.read_text())["techniques"]:
    for q in technique["questions"]:
        words = bare[f"bullshit_v1_{q['id']}"].split()
        for n in (2, 4, 8, 16):
            entries.append({"user_msg": q["question"] + SUFFIX, "suffix": " ".join(words[:n]), "source": f"bullshit_v1_{q['id']}", "words": n})
assert len(entries) == 220
OUT.write_text(json.dumps(entries, indent=1) + "\n")
print(f"wrote {OUT} ({len(entries)} entries)")
