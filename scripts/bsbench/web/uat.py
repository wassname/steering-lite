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
    curve_points = sum(len(c["points"]) for c in data["curves"] if c["method"] in data["shown"])
    drawn = page.locator("circle.mark, path.mark.end").count()
    print(f"curve points in points.json={curve_points} drawn={drawn}")
    assert drawn == curve_points, "SVG curve points differ from points.json"
    png_marks = json.loads((site / "plot_marks.json").read_text())
    print(f"PNG frontier marks (plot_marks.json from results.py)={png_marks['frontier_marks']} page drawn={drawn} methods png={png_marks['methods']} page={data['shown']}")
    assert png_marks["frontier_marks"] == drawn and png_marks["methods"] == data["shown"], "page and PNG draw different points"
    rows = page.locator("table tbody tr").count()
    assert rows == len(data["summary"]), (rows, len(data["summary"]))
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
