"""Refresh measured dose dots without changing cached scores or ratings. PI/OpenAI."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts/bsbench').resolve()))
import results

out = Path('outputs/bsbench/results/v5-9b-3seeds')
site = json.loads((out / 'points.json').read_text())
for curve in site['curves']:
    measured = results.method_curve(site['points'], curve['method'], curve['side'])
    curve['points'] = results.sweep(results.before_reversal(measured))
    curve['path'] = results.sweep_path(curve['points']) if measured else []
site['plot_labels'] = results.svg_labels(site)
(out / 'points.json').write_text(json.dumps(site, indent=1, allow_nan=False) + '\n')
title = f"steering-lite on Bullshit Bench v2: Qwen3.5-9B (full, 100 questions) — judge: Jev<br><sup>{site['setup'].split(' · ', 1)[1]}</sup>"
best = {(row['method'], side): point for row in site['summary'] for side, point in row['best'].items()}
figure = results.plot(site['points'], title, site['shown'], best)
figure.write_image(out / 'plot.png', width=1064, height=620, scale=2)
figure.write_html(out / 'plot.html', include_plotlyjs='cdn')
(out / 'plot_marks.json').write_text(json.dumps({'sweep_marks': sum(len(t.x) for t in figure.data if t.name == 'sweep'), 'methods': site['shown'], 'random_fills': sum(t.fill == 'toself' for t in figure.data)}) + '\n')
shutil.copy2(out / 'plot.png', 'assets/bsbench/qwen3.5-9b.png')
shutil.copy2(out / 'plot.png', Path(__file__).parent / 'plot.png')
print('DISPLAY_REFRESH measured dots; scores and ratings unchanged')
