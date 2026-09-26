"""Unanchored free-text description of what steering changed (PI/Claude, 2026-09-26).

The old DeepSeek blind judge wrote a free-text `change`, but its prompt also showed my 11 hand-written
labels, so the phrases are anchored to that list. Here the judge sees only the question and the two
answers: no labels, no target, no flaw, no mention of premises. Sample: up to N_PER questions per
(method, side, dose), at the blind-table doses (Pareto-best and strongest).
Output: outputs/bsbench/judgments/freetext_change.jsonl (one phrase per pair), input to cluster.py.

Run: cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-26_natural_labels/freetext.py
"""
import asyncio, hashlib, json, os, random, sys
sys.path.insert(0, ".")
import httpx
from loguru import logger
import results as R
from data import ROOT, default_model_dir, demo_rows, walk_certificates

MODEL = "deepseek/deepseek-v4-flash-0731"
N_PER = 60
OUT = ROOT / "outputs/bsbench/judgments/freetext_change.jsonl"
PROMPT = """Two answers to the same question came from the same assistant. Answer B was produced after an unknown change to the assistant.

In at most 8 words, how does B differ from A? Reply with the phrase only.

Question:
{question}

Answer A:
{a}

Answer B:
{b}"""

md = default_model_dir()
points = R.build_points(md, "full", set())
rows = {(r["method"], r["seed"], r["C"], r["side"], r["vignette"]): r for c in walk_certificates(md, "full") for r in demo_rows(md, c)}
rng = random.Random(0)
cells = {}
for method, (_, best, strongest) in R.choose(points).items():
    for kind, chosen in (("best", best), ("strongest", strongest)):
        for side, point in chosen.items():
            if point is None:
                continue
            qs = point["questions"]
            for q in rng.sample(qs, min(N_PER, len(qs))):
                seed = q["seed"]
                row = rows[method, seed, point["C"], side, q["scenario"]]
                k = hashlib.sha256(json.dumps([MODEL, PROMPT, row["prompt"], row["bare"], row["steered"]]).encode()).hexdigest()
                cells[k] = {"key": k, "method": method, "seed": seed, "C": point["C"], "side": side, "dose": kind, "vignette": q["scenario"], "row": row}
have = {json.loads(line)["key"] for line in OUT.open()} if OUT.exists() else set()
todo = [c for k, c in cells.items() if k not in have]
logger.info("FREETEXT cells={} cached={} todo={}", len(cells), len(cells) - len(todo), len(todo))


async def main():
    sem, cost = asyncio.Semaphore(16), 0.0
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}

    async def one(client, cell):
        nonlocal cost
        row = cell["row"]
        body = {"model": MODEL, "temperature": 0, "max_tokens": 40, "reasoning": {"enabled": False},
                "messages": [{"role": "user", "content": PROMPT.format(question=row["prompt"], a=row["bare"], b=row["steered"])}]}
        async with sem:
            for attempt in range(6):
                r = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, timeout=90)
                if r.status_code in (408, 429, 500, 502, 503, 504, 529):
                    await asyncio.sleep(5 * 2**attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            else:
                raise RuntimeError(f"freetext {cell['key']} failed 6 times")
        text = data["choices"][0]["message"]["content"].strip()
        cost += data["usage"].get("cost", 0)
        record = {k: v for k, v in cell.items() if k != "row"} | {"change": text, "model": data.get("model")}
        with OUT.open("a") as f:
            f.write(json.dumps(record) + "\n")

    async with httpx.AsyncClient() as client:
        for start in range(0, len(todo), 400):
            await asyncio.gather(*(one(client, c) for c in todo[start:start + 400]))
            logger.info("freetext progress={}/{} cost=${:.4f}", min(start + 400, len(todo)), len(todo), cost)
    logger.info("FREETEXT_COMPLETE cost=${:.4f}", cost)

asyncio.run(main())
