"""Check planned lower-dose coverage against existing certificates, without launching jobs. PI/OpenAI."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts/bsbench').resolve()))
import walk

root = Path('outputs/bsbench/Qwen--Qwen3.5-9B-g2351502a')
by_method = {}
new_doses = 0
for path in sorted((root / 'walks').glob('*_full.json')):
    cert = json.loads(path.read_text())
    flags = [cert['method'], '--seed', str(cert['seed']), '--cohort', 'full', '--preset', 'qwen3.5-9b', '--pairs', 'bsbench_v1']
    if cert['controls']:
        flags.append('--controls')
    args = walk.parse_args(flags)
    assert walk.walk_done(cert, args, require_bridges=False), path
    if cert['method'] in walk.PROMPT_METHODS:
        assert walk.walk_done(cert, args)
        continue
    assert not walk.walk_done(cert, args), path
    planned = {side: walk.bridge_doses(cert['start'][side]) for side in ('-C', '+C')}
    assert all(0 < min(ds) < max(ds) < cert['start'][side] for side, ds in planned.items())
    new_doses += sum(len(ds) for ds in planned.values())
    by_method.setdefault(cert['method'], []).append(planned)
    print(path.name, planned)
for method, plans in sorted(by_method.items()):
    for side in ('-C', '+C'):
        shared = set.intersection(*(set(p[side]) for p in plans))
        if method != 'random':
            assert shared, (method, side)
        print(f'SHARED {method} {side}: {sorted(shared)}')
assert new_doses == 332
print(f'PLAN_PASS new_doses={new_doses}; all learned-method sides gain >=1 shared lower dose; prompt unchanged')
