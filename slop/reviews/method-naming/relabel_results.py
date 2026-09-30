"""Relabel cached reports without rerunning bootstrap or judging. PI/OpenAI."""

import json
import re
import shutil
import sys
from pathlib import Path

from migrate_artifacts import NAMES, metadata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bsbench"))
import results

DISPLAY = {"VJP-delta": "VJP-resid", "VJP-cache": "VJP-value", "KV-cache Gram": "Value Gram"}
SUBS = NAMES | DISPLAY
PATTERN = re.compile("|".join(map(re.escape, sorted(SUBS, key=len, reverse=True))))
ASSETS = {"full": "bsbench_qwen3.5-4b_full.png", "27b-full": "bsbench_qwen3.5-27b_full.png", "olmo-full": "bsbench_olmo-2-32b_full.png"}

for out in sorted((ROOT / "outputs/bsbench/results").iterdir()):
    if not (out / "points.json").exists():
        continue
    original = json.loads((out / "points.json").read_text())
    site = metadata(original)
    assert all(a["questions"] == b["questions"] for a, b in zip(original["points"], site["points"], strict=True)), "answer/rating changed"
    for key in ("summary", "curves", "blind"):
        for before, after in zip(original[key], site[key], strict=True):
            assert {k: v for k, v in before.items() if k != "method"} == {k: v for k, v in after.items() if k != "method"}, f"statistics changed: {key}"
    (out / "points.json").write_text(json.dumps(site, indent=1, allow_nan=False) + "\n")
    for name in ("index.md", "plot.html"):
        path = out / name
        path.write_text(PATTERN.sub(lambda m: SUBS[m[0]], path.read_text()))
    marks = out / "plot_marks.json"
    marks.write_text(json.dumps(metadata(json.loads(marks.read_text()))) + "\n")
    results.COLORS.update(site["colors"])
    results.LABELS.update({m: m for m in site["colors"] if m not in results.LABELS})
    model = site["model_dir"].rsplit("-g", 1)[0].split("--")[-1]
    title = f"steering-lite on Bullshit Bench v2: {model} ({site['cohort']}, {len(site['questions'])} questions) — judge: Jev"
    best = {(r["method"], side): p for r in site["summary"] for side, p in r["best"].items()}
    fig = results.plot(site["points"], title, site["shown"], best)
    fig.write_image(out / "plot.png", width=1064, height=590, scale=2)
    html_path = out / "plot.html"
    page = html_path.read_text()
    figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn", default_width="100%", config={"responsive": True})
    html_path.write_text(page[:page.index("<div")] + figure_html + page[page.index("<pre>"):])
    if out.name in ASSETS:
        shutil.copy2(out / "plot.png", ROOT / "assets" / ASSETS[out.name])
    print(f"{out.name}: names updated; scores, CIs, curves, answers and ratings unchanged", flush=True)
