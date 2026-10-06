"""Refresh presentation from cached scores without new ratings or bootstrap draws. PI/OpenAI."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts/bsbench').resolve()))
import results

out = Path('outputs/bsbench/results/v5-9b-3seeds')
site = json.loads((out / 'points.json').read_text())
site['colors'] = results.COLORS
site['plot_labels'] = results.svg_labels(site)
(out / 'points.json').write_text(json.dumps(site, indent=1, allow_nan=False) + '\n')
title = f"steering-lite on Bullshit Bench v2: Qwen3.5-9B (full, 100 questions) — judge: Jev<br><sup>{site['setup'].split(' · ', 1)[1]}</sup>"
best = {(r['method'], side): p for r in site['summary'] for side, p in r['best'].items()}
figure = results.plot(site['points'], title, site['shown'], best)
figure.write_image(out / 'plot.png', width=1064, height=620, scale=2)
figure.write_html(out / 'plot.html', include_plotlyjs='cdn')
controls = results.control_plot(site['points'], site['shown'], '−C: detection or contrarianism? Qwen3.5-9B (full), 100 legitimate control questions')
controls.write_image(out / 'controls.png', width=1064, height=560, scale=2)
controls.write_html(out / 'controls.html', include_plotlyjs='cdn')
(out / 'plot_marks.json').write_text(json.dumps({'sweep_marks': sum(len(t.x) for t in figure.data if t.name == 'sweep'), 'methods': site['shown'], 'random_fills': sum(t.fill == 'toself' for t in figure.data)}) + '\n')
shutil.copy2(out / 'plot.png', 'assets/bsbench_qwen3.5-9b_main.png')
shutil.copy2(out / 'plot.png', Path(__file__).parent / 'plot_all_final.png')
report = (out / 'index.md').read_text().replace('at its coherent doses:', 'at its admissible doses:').replace('Reported, not scored.', 'These raw components feed the control-adjusted −C score.')
(out / 'index.md').write_text(report)
print('DISPLAY_REFRESH: scores unchanged; production plot() and svg_labels() used')
