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

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

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
    page.locator("select").last.select_option(question)  # the question picker (a judge switch may come first)
    blocks = page.locator(".answer").count()
    print(f"question={question} answer blocks={blocks}")
    assert blocks > 1
    assert data["questions"][1]["bare"][:40] in page.locator(".answer.bare").inner_text()

    page.locator("circle.mark").first.dispatch_event("click")
    assert page.locator(".answer.selected").count() == 1
    page.locator(".answer.selected").screenshot(path=str(site / "uat_selected.png"))
    page.screenshot(path=str(site / "uat_full.png"), full_page=True)
    if (site / "points_jev.json").exists():  # judge switch: the Jev view must draw exactly its own points and the Jev PNG's
        jev = json.loads((site / "points_jev.json").read_text())
        jev_marks = json.loads((site / "plot_marks_jev.json").read_text())
        jev_expect = sum(len(c["points"]) for c in jev["curves"] if c["method"] in jev["shown"])
        page.select_option("label.picker select", "jev")
        try:  # wait for points_jev.json to load and redraw; on timeout the assert below reports the counts
            page.wait_for_function(f"document.querySelectorAll('circle.mark, path.mark.end').length === {jev_expect}", timeout=15000)
        except PlaywrightTimeout:
            pass
        jev_drawn = page.locator("circle.mark, path.mark.end").count()
        print(f"JEV view: page drawn={jev_drawn} points_jev.json={jev_expect} PNG plot_marks_jev={jev_marks['frontier_marks']}")
        assert jev_drawn == jev_expect == jev_marks["frontier_marks"], "Jev view differs from points_jev.json or its PNG"
        page.screenshot(path=str(site / "uat_plot_jev.png"))
    assert not errors, errors
    print(f"UAT_PASS {url} screenshots: uat_plot.png uat_selected.png uat_full.png")
    browser.close()
server.shutdown()
