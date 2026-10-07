"""Verify downloaded low-dose additions against pre-run files. PI/OpenAI."""
import hashlib
import json
import math
from pathlib import Path

root = Path('outputs/bsbench/Qwen--Qwen3.5-9B-g2351502a')
evidence = Path('slop/reviews/2026-10-07_low_dose')
hashes = {path: digest for digest, path in (line.split(maxsplit=1) for line in (evidence / 'pre_backfill_sha256.txt').read_text().splitlines())}
paths = sorted((root / 'walks').glob('*_full.json'))
assert len(paths) == 86, len(paths)
changed = {}
shared = {}
new_answers = new_controls = doses = 0
seconds = 0.0
for path in paths:
    cert = json.loads(path.read_text())
    if cert['method'] == 'prompting':
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)]
        continue
    info = cert['low_dose_backfill']
    archive = root / info['previous_certificate']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == hashes[str(path)], archive
    old = json.loads(archive.read_text())
    changed[str(path)] = str(archive)
    expected = json.loads(archive.read_text())
    expected['sides'] = cert['sides']
    expected['timing']['total_s'] += info['seconds']
    expected['low_dose_backfill'] = info
    assert cert == expected, path
    seconds += info['seconds']
    for side, records in cert['sides'].items():
        additions = [r for r in records if r.get('bridge', False)]
        assert [r for r in records if not r.get('bridge', False)] == old['sides'][side], (path, side)
        anchor = 2 ** math.floor(math.log2(old['start'][side]))
        coefficients = [r['coefficient'] for r in additions]
        assert coefficients == [anchor / 4, anchor / 2], (path, side)
        assert info['coefficients'][side] == coefficients
        assert max(coefficients) < old['start'][side]
        shared.setdefault((cert['method'], side), []).append(set(coefficients))
        for record in additions:
            answers = [json.loads(line) for line in (root / record['answers']).read_text().splitlines()]
            assert len(answers) == len({a['scenario'] for a in answers}) == 100
            assert str(root / record['answers']) not in hashes
            new_answers += len(answers)
            if cert['controls']:
                controls = [json.loads(line) for line in (root / record['control_answers']).read_text().splitlines()]
                assert len(controls) == len({a['scenario'] for a in controls}) == 100
                assert {a['scenario'] for a in answers} == {a['scenario'] for a in controls}
                assert str(root / record['control_answers']) not in hashes
                new_controls += len(controls)
            doses += 1
    print(f"PRESERVED {path.name} added=4 seconds={info['seconds']:.3f}", flush=True)
assert len(changed) == 83
assert (doses, new_answers, new_controls) == (332, 33200, 25200)
for (method, side), values in sorted(shared.items()):
    common = set.intersection(*values)
    if method != 'random':
        assert common, (method, side)
    print(f'SHARED {method} {side}: {sorted(common)}', flush=True)
for name, digest in hashes.items():
    source = Path(changed.get(name, name))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, source
print(f'BACKFILL_PASS walks=83 doses={doses} new_answers={new_answers} new_controls={new_controls} original_files={len(hashes)} unchanged_or_exactly_archived; seconds={seconds:.3f}; GPU_USD_at_2.10_per_hour={seconds / 3600 * 2.10:.4f}', flush=True)
