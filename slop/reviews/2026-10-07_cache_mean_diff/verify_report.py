"""Verify cache-method inclusion and record dose-level aggregates. PI/OpenAI."""
import json
from pathlib import Path

root = Path('outputs/bsbench/results/v5-9b-3seeds')
out = Path('slop/reviews/2026-10-07_cache_mean_diff')
site = json.loads((root / 'points.json').read_text())
points = [p for p in site['points'] if p['method'] == 'cache_mean_diff']
assert len(site['points']) == 2732 and len(points) == 116
assert {p['seed'] for p in points} == {0, 1, 2}
assert all('control_claims' in p for p in points)
row = next(r for r in site['summary'] if r['method'] == 'cache_mean_diff')
assert row['seeds'] == 3 and row['score'] is not None
curves = [c for c in site['curves'] if c['method'] == 'cache_mean_diff']
assert len(curves) == 2 and all(c['points'] for c in curves)
assert all(c['points'][0]['C'] == 2 and c['points'][1]['C'] == 4 for c in curves)
for curve in curves:
    for dose in curve['tested']:
        at = [p for p in points if p['C'] == dose['C'] and p['side'] == curve['side']]
        assert len(at) == 3 and all(p['admissible'] for p in at)
        assert abs(sum(p['effect'] for p in at) / 3 - dose['effect']) < 1e-10
assert Path('assets/bsbench/qwen3.5-9b.png').read_bytes() == (root / 'plot.png').read_bytes()
keys = ('seed', 'side', 'C', 'effect', 'premise_effect', 'off_axis', 'admissible', 'control_claims', 'control_claims_bare')
artifact = {'summary': row, 'curves': curves, 'points': [{k: p[k] for k in keys} for p in points]}
(out / 'metrics.json').write_text(json.dumps(artifact, indent=2) + '\n')
print('CACHE_REPORT_PASS seeds=3 points=116 common_curves=2 lower_doses=2,4 controls_present; README asset matches plot')
print(json.dumps(row))
for curve in curves:
    sign = 1 if curve['side'] == '+C' else -1
    peak = max(curve['tested'], key=lambda p: sign * p['effect'])
    print(f"MAX_DIRECTED {curve['side']}: " + json.dumps(peak))
