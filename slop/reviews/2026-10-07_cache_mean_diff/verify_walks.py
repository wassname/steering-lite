"""Check cache-method walks and retain mechanically selected raw examples. PI/OpenAI."""
import json
import math
from pathlib import Path

root = Path('outputs/bsbench/Qwen--Qwen3.5-9B-g2351502a')
out = Path('slop/reviews/2026-10-07_cache_mean_diff')
rows = []
seconds = 0.0
samples = ['# Cache-method answer samples\n\nPI/OpenAI. First benchmark scenario in each answer file, both signs, lowest/middle/highest measured coefficient for each seed. No choice by judge score.\n']
bare = [json.loads(line) for line in (root / 'answers/bare/bare.jsonl').read_text().splitlines()]
scenario = bare[0]['scenario']
samples += ['## Bare\n', bare[0]['prompt'], '\n\n' + bare[0]['text'] + '\n']
for seed in range(3):
    cert_path = root / 'walks' / f'cache_mean_diff_s{seed}_full.json'
    cert = json.loads(cert_path.read_text())
    assert cert['status'] == 'COMPLETE' and cert['controls']
    assert cert['method'] == 'cache_mean_diff' and cert['seed'] == seed
    assert cert['gen']['pairs'] == 'bsbench_v1'
    seconds += cert['timing']['total_s']
    for side, points in cert['sides'].items():
        coefficients = [p['coefficient'] for p in points]
        anchor = 2 ** math.floor(math.log2(cert['start'][side]))
        assert coefficients[:2] == [anchor / 4, anchor / 2]
        assert cert['state'][side]['done']
        examples = {0, len(points) // 2, len(points) - 1}
        for i, point in enumerate(points):
            answers = [json.loads(line) for line in (root / point['answers']).read_text().splitlines()]
            controls = [json.loads(line) for line in (root / point['control_answers']).read_text().splitlines()]
            assert len(answers) == len(controls) == 100
            assert {a['scenario'] for a in answers} == {a['scenario'] for a in controls} == {a['scenario'] for a in bare}
            if i in examples:
                answer = next(a for a in answers if a['scenario'] == scenario)
                control = next(a for a in controls if a['scenario'] == scenario)
                samples += [f'\n## Seed {seed}, {side}, C={point["coefficient"]:g}\n', answer['text'], '\n\nControl question: ' + control['prompt'], '\n\n' + control['text'] + '\n']
        rows.append({'seed': seed, 'side': side, 'c0': cert['c0'][side], 'start': cert['start'][side], 'points': len(points), 'low_doses': coefficients[:2], 'last_dose': coefficients[-1]})
        print(json.dumps(rows[-1]))
(out / 'answer_samples.md').write_text('\n'.join(samples))
(out / 'walk_summary.json').write_text(json.dumps({'rows': rows, 'total_s': seconds, 'gpu_usd_at_2_10': seconds / 3600 * 2.10}, indent=2) + '\n')
print(f'CACHE_WALKS_PASS seeds=3 controls=100 lower_points=12 dose_points={sum(r["points"] for r in rows)} seconds={seconds:.3f} estimated_GPU_USD={seconds / 3600 * 2.10:.4f}')
