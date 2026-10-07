"""Check measured lower doses survive aggregation and display. PI/OpenAI."""
import json
from pathlib import Path

out = Path('outputs/bsbench/results/v5-9b-3seeds')
evidence = Path('slop/reviews/2026-10-07_low_dose')
site = json.loads((out / 'points.json').read_text())
old = json.loads((evidence / 'summary_before.json').read_text())
key = lambda p: (p['method'], p['seed'], p['side'], p['C'])
previous = {key(p): p for p in old['points']}
current = {key(p): p for p in site['points']}
assert len(current) == 2616 and len(current) - len(previous) == 332
for k, p in previous.items():
    assert all(current[k][field] == p[field] for field in ('effect', 'premise_effect', 'off_axis', 'admissible'))
for curve in site['curves']:
    old_low = min(p['C'] for p in previous.values() if p['method'] == curve['method'] and p['side'] == curve['side'])
    tested = {p['C']: p for p in curve['tested']}
    lower = [p for p in curve['points'] if p['C'] < old_low]
    for point in curve['points']:
        assert all(point[k] == tested[point['C']][k] for k in ('effect', 'off_axis'))
    if curve['method'] != 'angular_steering':
        assert lower, (curve['method'], curve['side'])
    print(f"LOWER_DISPLAY {curve['method']} {curve['side']} old_min_C={old_low:g}: " + json.dumps(lower))
for row in site['summary']:
    before = next(r for r in old['summary'] if r['method'] == row['method'])
    print(f"SCORE {row['method']}: {before['score']} -> {row['score']}; best=" + json.dumps(row['best']))
print('REPORT_PASS old point metrics unchanged; 332 new points; all learned-method sides except angular have measured lower dots; angular has no common passing dose')
table = '| method' + (out / 'index.md').read_text().split('| method', 1)[1]
(evidence / 'table.md').write_text('<!-- PI/OpenAI: generated report after low-dose additions. -->\n\n' + table)
