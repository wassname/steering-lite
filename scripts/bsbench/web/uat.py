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
    print(f"PNG frontier marks (plot_marks.json from results.py)={png_marks['frontier_marks']} page drawn={drawn} methods png={png_marks['methods']} page={data['shown']}")
    assert png_marks["frontier_marks"] == drawn and png_marks["methods"] == data["shown"], "page and PNG draw different points"
    assert [z["percentile"] for z in data["zones"]] == [90, 75, 50]
    zones = page.locator(".zone")
    assert zones.count() == 3
    for i, zone in enumerate(data["zones"]):
        assert zones.nth(i).get_attribute("data-percentile") == str(zone["percentile"])
        assert zones.nth(i).evaluate("el => getComputedStyle(el).fill.replaceAll(' ', '') === el.getAttribute('fill')")
    if data["view"] == "prompt":
        assert set(data["shown"]) == {"prompting_scale", "prompting_engineered_scale", "mean_diff"}
        for method in data["shown"]:
            assert page.get_by_role("button", name=method, exact=True).get_attribute("aria-pressed") == "true"
    print("random regions: p90/p75/p50 with distinct fills; requested opening methods visible")
    for curve in data["curves"]:
        directed_path = [(x if curve["side"] == "+C" else -x) for x, y in curve["path"]]
        assert all(a <= b for a, b in zip(directed_path, directed_path[1:])), "Pareto path must not double back"
        seeds = {p["seed"] for p in data["points"] if p["method"] == curve["method"]}
        for mark in curve["points"]:
            at = [p for p in data["points"] if p["method"] == curve["method"] and p["side"] == curve["side"] and p["C"] == mark["C"]]
            assert {p["seed"] for p in at} == seeds and all(p["admissible"] for p in at), "curve includes a rejected dose"
    prompt_count = sum(p["method"] in ("prompting", "prompting_engineered") and p["admissible"] for p in data["points"])
    assert page.locator(".prompt-baseline").count() == prompt_count, "swept prompts must not appear as baseline stars"
    assert page.locator(".prompt-baseline path").evaluate_all("els => els.every(el => { const t = el.transform.baseVal.consolidate().matrix; return t.e >= 0 && t.e <= 1000 && t.f >= 0 && t.f <= 560; })"), "prompt baseline outside SVG view"
    gain_image = page.locator('img[src="prompt_gains.png"]')
    if any(p.get("fixed_grid", False) for p in data["points"]):
        assert gain_image.count() == 1
        page.wait_for_function("document.querySelector('img[src=\"prompt_gains.png\"]').naturalWidth > 0")
        assert not page.locator('img[src="prompt_gains_all.png"]').is_visible(), "rejected doses must be hidden by default"
        page.locator("#prompt-gains").screenshot(path=str(site / "uat_prompt_gains.png"))
    print(f"prompt baseline stars={prompt_count}; no rejected curve points, rejected stars, or swept-prompt stars")
    rows = page.locator("table:not(.blind) tbody tr").count()
    assert rows == len(data["summary"]), (rows, len(data["summary"]))
    blind_rows = page.locator("table.blind tbody tr").count()
    assert blind_rows == len(data["blind"]), (blind_rows, len(data["blind"]))
    # the label cell shows every label >= 2% from points.json, not only the top one
    cell = page.locator("table.blind tbody tr").first.locator("td").last.inner_text()
    assert cell.count("%") == sum(v >= 0.02 for v in data["blind"][0]["labels"].values()), cell
    page.evaluate("window.scrollTo(0, 0)")
    assert page.evaluate("window.scrollY") == 0
    page.screenshot(path=str(site / "uat_plot.png"), full_page=False)

    sweep_methods = {p["method"] for p in data["points"] if p.get("fixed_grid", False)}
    if sweep_methods:
        for curve in data["curves"]:
            if curve["method"] in sweep_methods:
                assert curve["path"][0] in [[p["effect"], p["off_axis"]] for p in curve["points"]], "prompt curve must start at a measured point, not bare"
        for method in set(data["shown"]) ^ sweep_methods:
            page.get_by_role("button", name=method, exact=True).click()
        expected = sum(len(c["points"]) for c in data["curves"] if c["method"] in sweep_methods)
        assert page.locator("circle.mark, path.mark.end").count() == expected
        assert page.locator(".prompt-baseline").count() == prompt_count
        page.screenshot(path=str(site / "uat_prompt_sweeps.png"), full_page=False)
        print(f"sweep-only curve marks={expected}; baseline stars still={prompt_count}")
        for method in set(data["shown"]) ^ sweep_methods:
            page.get_by_role("button", name=method, exact=True).click()

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
