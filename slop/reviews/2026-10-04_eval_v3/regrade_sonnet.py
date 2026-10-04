"""Is Jev tricked on the -C side? Re-grade answers with one of BullshitBench's own panel judges and its own prompt.

BullshitBench (petergpt/bullshit-benchmark @ 6f6e28b4, scripts/openrouter_benchmark.py) grades with a panel of
Claude Sonnet 4.6, GPT-5.2 and Gemini 3.1 Pro, integer 0/1/2, system prompt DEFAULT_JUDGE_SYSTEM_PROMPT and user
template DEFAULT_JUDGE_USER_TEMPLATE. Here: Sonnet 4.6 alone, same prompts, on 40 questions x (bare, prompt -C,
mean_diff -C at its scored dose, vjp_resid -C at its scored dose); compare with Jev's expected score. PI/OpenAI 2026-10-04.

  cd scripts/bsbench && PYTHONPATH=. uv run --extra benchmark python ../../slop/reviews/2026-10-04_eval_v3/regrade_sonnet.py
"""
import asyncio
import json
import os
import random
import re
from pathlib import Path
from statistics import mean

import httpx

from data import load_cohort, read_answers
from judge import BSB_SYSTEM, bsb_request, cached, key

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("regrade_sonnet.jsonl")
M = ROOT / "outputs/bsbench/Qwen--Qwen3.5-4B-g1f092bc2"
TEMPLATE = json.loads((ROOT / ".local/bsb_judge_text.json").read_text())["DEFAULT_JUDGE_USER_TEMPLATE"]
JUDGE = "anthropic/claude-sonnet-4.6"
CONDITIONS = {"bare": M / "answers/bare/bare.jsonl", "prompt -C": M / "answers/prompting_s0/-C_C1.jsonl",
              "mean_diff -C 0.5": M / "answers/mean_diff_s0/-C_C0.5.jsonl", "vjp_resid -C 0.198": M / "answers/vjp_resid_s0/-C_C0.1984251315.jsonl"}

questions = load_cohort()
sample = random.Random(20261004).sample(sorted(questions), 40)
done = {(r["condition"], r["scenario"]): r for r in map(json.loads, OUT.open())} if OUT.exists() else {}


async def grade(client, condition, scenario, text):
    q = questions[scenario]
    body = {"model": JUDGE, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [
        {"role": "system", "content": BSB_SYSTEM},
        {"role": "user", "content": TEMPLATE.format(question=q["prompt"], nonsensical_element=q["nonsensical_element"], response=text)}]}
    for attempt in range(5):
        try:
            r = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            out = json.loads(re.search(r"\{.*\}", content, re.S)[0])
            return {"condition": condition, "scenario": scenario, "score": int(out["score"]), "justification": out["justification"], "text": text}
        except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as error:
            print("retry", condition, scenario, repr(error)[:120])
            await asyncio.sleep(3 * 2**attempt)
    raise RuntimeError(f"{condition} {scenario} failed")


async def main():
    answers = {c: read_answers(p) for c, p in CONDITIONS.items()}
    todo = [(c, s) for c in CONDITIONS for s in sample if (c, s) not in done]
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient() as client:
        async def run(c, s):
            async with sem:
                row = await grade(client, c, s, answers[c][s]["text"])
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done[c, s] = row
        await asyncio.gather(*(run(c, s) for c, s in todo))
    have = cached()
    jev = lambda c, s: have[key(bsb_request(questions[s]["prompt"], questions[s]["nonsensical_element"], answers[c][s]["text"]))]["bs_score"]["score"]
    print("| condition | Sonnet 4.6 mean (0-2) | Jev mean (0-2) | Sonnet gain vs bare | Jev gain vs bare | Sonnet share 2 | |Jev − Sonnet| mean |")
    print("|---|---|---|---|---|---|---|")
    base_s = mean(done["bare", s]["score"] for s in sample)
    base_j = mean(jev("bare", s) for s in sample)
    for c in CONDITIONS:
        s_scores = [done[c, s]["score"] for s in sample]
        j_scores = [jev(c, s) for s in sample]
        print(f"| {c} | {mean(s_scores):.2f} | {mean(j_scores):.2f} | {mean(s_scores) - base_s:+.2f} | {mean(j_scores) - base_j:+.2f} | "
              f"{sum(x == 2 for x in s_scores) / len(s_scores):.0%} | {mean(abs(a - b) for a, b in zip(s_scores, j_scores)):.2f} |")
    print("\nLargest disagreements (Jev − Sonnet):")
    rows = sorted(((jev(c, s) - done[c, s]["score"], c, s) for c in CONDITIONS for s in sample), key=lambda r: -abs(r[0]))
    for d, c, s in rows[:8]:
        print(f"\n[{c} | {s}] Jev {jev(c, s):.2f} vs Sonnet {done[c, s]['score']}: {done[c, s]['justification']}\n  {answers[c][s]['text'][:300]!r}")


asyncio.run(main())
