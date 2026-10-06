"""Artifact checks for the completed 9B expansion; no GPU or API calls. PI/OpenAI."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).parent
model = root / 'outputs/bsbench/Qwen--Qwen3.5-9B-g2351502a'
report = root / 'outputs/bsbench/results/v5-9b-3seeds'
site = json.loads((report / 'points.json').read_text())
learned = {r['method'] for r in site['summary']} - {'random', 'prompting'}
assert len(learned) == 21
assert len(site['questions']) == 100
assert site['random_seeds'] == list(range(20))
seconds = defaultdict(float)
certs = []
for method in sorted(learned | {'random', 'prompting'}):
    seeds = range(20) if method == 'random' else range(3)
    for seed in seeds:
        path = model / f'walks/{method}_s{seed}_full.json'
        cert = json.loads(path.read_text())
        assert cert['status'] == 'COMPLETE', path
        assert cert['method'] == method and cert['seed'] == seed
        assert cert['gen']['pairs'] == 'bsbench_v1'
        assert cert['controls'] == (method != 'random')
        assert all(cert['sides'][s] for s in ('-C', '+C'))
        seconds[method] += cert['timing']['total_s']
        certs.append({'file': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'total_s': cert['timing']['total_s']})
for point in site['points']:
    assert len(point['questions']) == 100
    assert 0 <= point['off_axis'] <= 4
    assert point['admissible'] == (point['off_axis'] <= site['max_off_axis'])
    if point['method'] != 'random':
        assert 'control_claims' in point
    expected = point['premise_effect']
    if point['side'] == '-C' and point['method'] != 'random':
        expected += 3 * (point['control_claims'] - point['control_claims_bare'])
    assert abs(point['effect'] - expected) < 1e-8
    assert all(q['scenario'] in {r['scenario'] for r in site['questions']} for q in point['questions'])
print(f"PASS: {len(learned)} learned methods x 3 seeds; prompt x 3; random x 20; {len(certs)} COMPLETE certificates")
print(f"PASS: {len(site['points'])} dose/seed/side points x 100 questions; off-axis nonnegative; control formula reproduced")
print('NOTE: COMPLETE means the walk finished, not that every method has an admissible score')
new_methods = learned - {'mean_diff', 'vjp_resid'}
gpu = sum(seconds[m] for m in new_methods) / 3600 * 2.10
print(f"Expansion completed GPU time: {sum(seconds[m] for m in new_methods):.3f} seconds x $2.10/hour = ${gpu:.4f}")
judge = 0
for name in ('judge.log', 'judge_corda.log'):
    log = (evidence / name).read_text()
    assert 'JUDGE_COMPLETE missing=0' in log
    for line in log.splitlines():
        match = re.search(r'jev progress=(\d+)/(\d+) cost=\$([\d.]+)', line)
        if match and match[1] == match[2]:
            judge += float(match[3])
            print(f'{name}: {line}')
print(f"Expansion logged Jev cost: ${judge:.4f}; combined accounted estimate: ${gpu + judge:.4f}")
print('Cost is completed walk wall time at GPU rate plus logged API cost, not an invoice; failed attempt, setup/CPU/memory charges and overwritten retries are excluded.')
(evidence / 'certificate_inventory.json').write_text(json.dumps(certs, indent=2) + '\n')
for method, value in sorted(seconds.items(), key=lambda item: -item[1]):
    print(f'GPU_TIME {method}: {value:.3f}s ${value / 3600 * 2.1:.4f}')
text = ['# Fixed-selection answer inspection', '', 'Author PI/OpenAI. First benchmark question, seed 0, lowest tested dose on each side for every method. This selection is fixed without inspecting answers. Full outputs, not selected excerpts.', '']
scenario = site['questions'][0]
text += [f"Question: {scenario['prompt']}", f"Known flaw: {scenario['flaw']}", f"Bare answer: {scenario['bare']}", '']
for method in sorted(learned | {'random', 'prompting'}):
    for side in ('-C', '+C'):
        point = min((p for p in site['points'] if p['method'] == method and p['seed'] == 0 and p['side'] == side), key=lambda p: p['C'])
        q = next(q for q in point['questions'] if q['scenario'] == scenario['scenario'])
        text += [f"## {method} {side}, C={point['C']}", f"Mean raw premise change {point['premise_effect']:+.4f}; off-axis {point['off_axis']:.4f}", '', q['text'], '']
(evidence / 'answer_samples.md').write_text('\n'.join(text))
(evidence / 'table_final.md').write_text((report / 'index.md').read_text())
print('Artifacts: certificate_inventory.json, answer_samples.md, table_final.md')
