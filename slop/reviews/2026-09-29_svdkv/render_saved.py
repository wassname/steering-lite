"""Render saved statistics with fixed sink-method colors, without rejudging or resampling. PI/OpenAI."""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bsbench"))
import results

for cohort in ("full", "dev"):
    out = ROOT / "outputs/bsbench/results" / cohort
    site = json.loads((out / "points.json").read_text())
    results.COLORS.update(site["colors"])
    results.LABELS.update({m: m for m in site["colors"] if m not in results.LABELS})
    title = f"steering-lite on Bullshit Bench v2: Qwen3.5-4B ({cohort}, {len(site['questions'])} questions) — judge: Jev"
    best = {(r["method"], side): p for r in site["summary"] for side, p in r["best"].items()}
    fig = results.plot(site["points"], title, site["shown"], best)
    fig.write_image(out / "plot.png", width=1064, height=590, scale=2)
    text = (out / "plot.html").read_text()
    start, end = text.index("<div"), text.index("<pre>")
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn", default_width="100%", config={"responsive": True})
    (out / "plot.html").write_text(text[:start] + fig_html + text[end:])
    expected = json.loads((out / "plot_marks.json").read_text())["frontier_marks"]
    assert sum(len(t.x) for t in fig.data if t.name == "frontier") == expected
    print(f"rendered {cohort}; scores, intervals, curves and selections unchanged", flush=True)
shutil.copyfile(ROOT / "outputs/bsbench/results/full/plot.png", ROOT / "assets/bsbench_qwen3.5-4b_full.png")
