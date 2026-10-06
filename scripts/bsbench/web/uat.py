"""Browser check of the built results page: it loads points.json, draws every curve point, and the
explorer shows answers for a picked question and a clicked plot point. Writes screenshots.

    uv run --with playwright python scripts/bsbench/web/uat.py outputs/bsbench/results/dev
"""

import functools
import http.server
import json
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

site = Path(sys.argv[1]).resolve()
data = json.loads((site / "points.json").read_text())
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/index.html"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_selector("svg")
    assert f"{data['cohort'].upper()} · {len(data['questions'])} questions" in page.locator("svg").text_content()
    curve_points = sum(len(c["points"]) for c in data["curves"] if c["method"] in data["shown"])
    drawn = page.locator("circle.mark, path.mark.end").count()
    print(f"curve points in points.json={curve_points} drawn={drawn}")
    assert drawn == curve_points, "SVG curve points differ from points.json"
    png_marks = json.loads((site / "plot_marks.json").read_text())
    print(f"PNG sweep marks (plot_marks.json from results.py)={png_marks['sweep_marks']} page drawn={drawn} methods png={png_marks['methods']} page={data['shown']}")
    assert png_marks["sweep_marks"] == drawn and png_marks["methods"] == data["shown"], "page and PNG draw different points"
    assert page.locator("circle.sample, .best").count() == 0, "faint dots and score rings are not drawn by default"
    assert page.locator("path.mark.end").count() == sum(bool(c["points"]) for c in data["curves"] if c["method"] in data["shown"]), "one x per drawn line"
    assert page.locator(".curve-line").count() == sum(bool(c["points"]) for c in data["curves"] if c["method"] in data["shown"])
    for curve in data["curves"]:
        assert all((x is None) == (y is None) for x, y in curve["path"])
    for label in data["plot_labels"]:
        drawn_label = page.locator(f'.curve-label[data-method="{label["method"]}"][data-side="{label["side"]}"] text')
        assert drawn_label.text_content() == label["text"], "default-view curves need labels"
        assert drawn_label.evaluate("el => { const b = el.getBBox(); return b.x >= 70 && b.x+b.width <= 980 && b.y >= 30 && b.y+b.height <= 510; }")
    assert f"random: {len(data['random_seeds'])} directions" in page.locator("svg").text_content()
    assert [z["percentile"] for z in data["zones"]] == [90, 75, 50]
    assert data["admissibility"] == "jev_mean_off_axis"
    assert all(p["admissible"] == (p["off_axis"] <= data["max_off_axis"]) for p in data["points"]), "only Jev's pairwise off-axis change decides admissibility"
    explanation = page.locator("body").text_content()
    assert "discrete observed ranks" in explanation and "min–max" in explanation, "small-sample bands must not imply precise percentile bounds"
    assert "individual retained answers can still be badly damaged" in explanation, "passing means do not certify each answer"
    zones = page.locator(".zone")
    assert zones.count() == 3
    for i, zone in enumerate(data["zones"]):
        assert zones.nth(i).get_attribute("data-percentile") == str(zone["percentile"])
    if data["view"] == "user":
        assert "User-turn steering" in page.locator("h1").text_content()
        assert "only while the model reads the user's message" in explanation
        assert not any(p["method"].endswith("-user") for p in data["points"]), "user view renames <method>-user to <method>"
    print("random regions: p90/p75/p50 with distinct fills; requested opening methods visible")
    for curve in data["curves"]:
        assert [p["C"] for p in curve["points"]] == sorted(p["C"] for p in curve["points"]), "sweep must be in dose order"
        if curve["points"]:
            assert {p["C"] for p in curve["points"]} <= {p["C"] for p in curve["tested"]}, "line drawn only through passing doses"
            assert curve["path"][-1] == [curve["points"][-1]["effect"], curve["points"][-1]["off_axis"]], "line pinned at the x"
        seeds = {p["seed"] for p in data["points"] if p["method"] == curve["method"]}
        for mark in curve["tested"]:
            at = [p for p in data["points"] if p["method"] == curve["method"] and p["side"] == curve["side"] and p["C"] == mark["C"]]
            assert {p["seed"] for p in at} == seeds and all(p["admissible"] for p in at), "curve includes a rejected dose"
    prompt_count = len({(p["method"], p["side"]) for p in data["points"] if p["method"] in ("prompting", "prompting_engineered")})  # one star per prompt and side
    assert page.locator(".prompt-baseline").count() == prompt_count, "one star per plain prompt"
    assert page.locator(".prompt-baseline path").evaluate_all("els => els.every(el => { const t = el.transform.baseVal.consolidate().matrix; return t.e >= 0 && t.e <= 1000 && t.f >= 0 && t.f <= 560; })"), "prompt baseline outside SVG view"
    print(f"prompt baseline stars={prompt_count}; no rejected curve points, rejected stars, or swept-prompt stars")
    rows = page.locator("table.summary tbody tr").count()
    assert rows == len(data["summary"]), (rows, len(data["summary"]))
    blind_rows = page.locator("table.blind tbody tr").count()
    assert blind_rows == len(data["blind"]), (blind_rows, len(data["blind"]))
    # the label cell shows every label >= 2% from points.json, not only the top one
    cell = page.locator("table.blind tbody tr").first.locator("td").last.inner_text()
    assert cell.count("%") == sum(v >= 0.02 for v in data["blind"][0]["labels"].values()), cell
    page.evaluate("window.scrollTo(0, 0)")
    assert page.evaluate("window.scrollY") == 0
    page.screenshot(path=str(site / "uat_plot.png"), full_page=False)

    question = data["questions"][1]["scenario"]
    page.select_option("select", question)
    blocks = page.locator(".answer").count()
    print(f"question={question} answer blocks={blocks}")
    assert blocks > 1
    assert data["questions"][1]["bare"][:40] in page.locator(".answer.bare").inner_text()
    blind = page.locator("p.judge", has_text="blind judge").count()
    print(f"blind judge lines={blind} (default blocks are Pareto-best doses, which judge.py rates blind)")
    assert blind > 0

    page.locator("circle.mark").first.dispatch_event("click")
    assert page.locator(".answer.selected").count() == 1
    page.locator(".answer.selected").screenshot(path=str(site / "uat_selected.png"))
    page.screenshot(path=str(site / "uat_full.png"), full_page=True)
    assert not errors, errors
    print(f"UAT_PASS {url} screenshots: uat_plot.png uat_selected.png uat_full.png")
    browser.close()
server.shutdown()
