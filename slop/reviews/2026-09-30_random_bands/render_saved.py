"""Redraw cached reports without rerunning judges or bootstrap. Adapted from the sink report renderer. -- PI/OpenAI"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bsbench"))
import results

assets = {"full": "bsbench_qwen3.5-4b_full.png", "27b-full": "bsbench_qwen3.5-27b_full.png", "olmo-full": "bsbench_olmo-2-32b_full.png"}
evidence = {}
for report in ("prompt-dev", "dev", "full", "27b-full", "olmo-full"):
    out = ROOT / "outputs/bsbench/results" / report
    source = "prompt-dev" if report == "dev" else report
    original = json.loads((ROOT / ".local/random-bands" / f"{source}.json").read_text())
    site = json.loads(json.dumps(original))
    site["view"] = "prompt" if report == "prompt-dev" else "benchmark"
    if report == "prompt-dev":
        site["shown"] = ["prompting_scale", "prompting_engineered_scale", "mean_diff"]
    elif report == "dev":
        eligible = [r["method"] for r in site["summary"] if r["method"] not in ("random", *results.PROMPTS) and "-" not in r["method"] and r["score"] is not None]
        site["shown"] = eligible[:results.TOP_N_PLOT]
        site["shown"] += [m for m in ("prompting_scale", "prompting_engineered_scale") if m not in site["shown"]]
    else:
        for p in site["points"]:
            p["fixed_grid"] = False
    site["zones"] = results.random_zones(site["points"])
    assert site["zones"][0]["bounds"] == [tuple(row) for row in original["zone"]], "outer empirical bounds changed"
    del site["zone"]
    for rows in zip(*(z["bounds"] for z in site["zones"]), strict=True):
        assert len({r[1] for r in rows}) == 1
        lo = [min(0, r[2]) for r in rows]
        hi = [max(0, r[3]) for r in rows]
        assert lo[0] <= lo[1] <= lo[2] <= 0 <= hi[2] <= hi[1] <= hi[0]
    half = len(site["zones"][0]["path"]) // 2
    edges = [list(zip(z["path"][:half], reversed(z["path"][half:]), strict=True)) for z in site["zones"]]
    for row in zip(*edges, strict=True):
        assert len({p[1] for edge in row for p in edge}) == 1
        assert row[0][0][0] <= row[1][0][0] <= row[2][0][0] <= 0 <= row[2][1][0] <= row[1][1][0] <= row[0][1][0]
    for curve in site["curves"]:
        full = results.method_curve(site["points"], curve["method"], curve["side"])
        curve["path"] = results.smooth_path(results.frontier(full), curve["side"]) if full else []
    results.COLORS.update(site["colors"])
    results.LABELS.update({m: m for m in site["colors"] if m not in results.LABELS})
    model = site["model_dir"].rsplit("-g", 1)[0].split("--")[-1]
    heading = "Prompt embedding sweeps vs mean difference" if site["view"] == "prompt" else "steering-lite on Bullshit Bench v2"
    title = f"{heading}: {model} ({site['cohort']}, {len(site['questions'])} questions) — judge: Jev"
    best = {(r["method"], side): p for r in site["summary"] for side, p in r["best"].items()}
    fig = results.plot(site["points"], title, site["shown"], best)
    for trace, zone in zip(fig.data[:3], site["zones"], strict=True):
        assert list(trace.x) == [p[0] for p in zone["path"]] and list(trace.y) == [p[1] for p in zone["path"]]
    fig.write_image(out / "plot.png", width=1064, height=590, scale=2)
    text = (out / "plot.html").read_text()
    start, end = text.index("<div"), text.index("<pre>")
    (out / "plot.html").write_text(text[:start] + fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True}) + text[end:])
    marks = sum(len(t.x) for t in fig.data if t.name == "frontier")
    (out / "plot_marks.json").write_text(json.dumps({"frontier_marks": marks, "methods": site["shown"]}) + "\n")
    for field in ("summary", "blind", "questions"):
        assert site[field] == original[field], f"{field} changed"
    assert [c["points"] for c in site["curves"]] == [c["points"] for c in original["curves"]], "markers changed"
    for before, after in zip(original["points"], site["points"], strict=True):
        assert all(after[k] == v for k, v in before.items())
    (out / "points.json").write_text(json.dumps(site, indent=1) + "\n")
    if report in ("prompt-dev", "dev"):
        gain = results.prompt_gain_plot(site["points"], f"Prompt embedding gains: {model} ({site['cohort']}, {len(site['questions'])} questions)")
        diagnostic = results.prompt_gain_plot(site["points"], f"Diagnostic — all tested gains: {model}", include_rejected=True)
        for fig_name, figure in (("prompt_gains", gain), ("prompt_gains_all", diagnostic)):
            figure.write_image(out / f"{fig_name}.png", width=1064, height=590, scale=2)
            figure.write_html(out / f"{fig_name}.html", include_plotlyjs="cdn")
        for trace in gain.data:
            method, side = next((m, s) for m in ("prompting_scale", "prompting_engineered_scale") for s in ("+C", "-C") if trace.name == f"{results.LABELS[m]} {s}")
            accepted = {p["C"] for p in results.method_curve(site["points"], method, side)}
            assert {float(x) for x, y in zip(trace.x, trace.y, strict=True) if y is not None} == accepted
            assert not trace.connectgaps
        print("GAIN_FILTER_PASS: rejected doses omitted; identical method_curve admissibility; no lines across failed gains", flush=True)
        if report == "dev":
            shutil.copyfile(ROOT / "outputs/bsbench/results/prompt-dev/index.md", out / "index.md")
        md = (out / "index.md").read_text().replace("![All tested prompt gains, including inadmissible doses](prompt_gains.png)", "![Admissible prompt gains](prompt_gains.png)\n\n[Diagnostic: all gains, including rejected doses](prompt_gains_all.html)")
        (out / "index.md").write_text(md)
    else:
        shutil.copyfile(out / "plot.png", ROOT / "assets" / assets[report])
    evidence[report] = {"shown": site["shown"], "doses": len(site["zones"][0]["bounds"]) - 1, "percentiles": [z["percentile"] for z in site["zones"]], "plot_sha256": hashlib.sha256((out / "plot.png").read_bytes()).hexdigest(), "checks": "old p90 bounds exact; nested zero-filled polygons; PNG/JSON paths exact; scores/intervals/selections/answers unchanged"}
    print(report, evidence[report], flush=True)
Path(__file__).with_name("verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
