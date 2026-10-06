"""Real tiny-model walk then historical-certificate backfill reproduction. PI/OpenAI."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts/bsbench').resolve()))
import walk

flags = ['mean_diff', '--smoke', '--preset', 'tiny-random', '--n-pairs', '4', '--extract-batch-size', '2',
         '--max-length', '256', '--layers', '1,2', '--target-layer', '4', '--max-rungs', '2',
         '--controls', '--tag', 'lowdose-smoke']
args = walk.parse_args(flags)
walk.configure(args)
root = walk.model_dir(args.model)
path = root / 'walks' / f'{args.name}_s0_dev.json'
command = [sys.executable, 'scripts/bsbench/walk.py', *flags]
subprocess.run(command, check=True)
full = json.loads(path.read_text())
assert walk.walk_done(full, args)
assert all(len(rungs) == 4 for rungs in full['sides'].values())
vector = root / 'vectors' / f'{args.vector_name}_s0.safetensors'
vector_hash = hashlib.sha256(vector.read_bytes()).hexdigest()
original_answers = {}
for side, rungs in full['sides'].items():
    bridges = [r for r in rungs if r.get('bridge', False)]
    assert [r['coefficient'] for r in bridges] == list(walk.bridge_doses(full['start'][side]))
    for rung in rungs:
        for key in ('answers', 'control_answers'):
            answer = root / rung[key]
            if rung in bridges:
                answer.unlink()
            else:
                original_answers[answer] = hashlib.sha256(answer.read_bytes()).hexdigest()
    full['sides'][side] = [r for r in rungs if r not in bridges]
path.write_text(json.dumps(full, indent=2) + '\n')
assert not walk.walk_done(full, args)
subprocess.run(command, check=True)
backfilled = json.loads(path.read_text())
assert walk.walk_done(backfilled, args)
assert backfilled['state'] == full['state'] and backfilled['start'] == full['start']
assert backfilled['timing']['total_s'] > full['timing']['total_s']
archive = json.loads((root / backfilled['low_dose_backfill']['previous_certificate']).read_text())
assert archive == full
assert hashlib.sha256(vector.read_bytes()).hexdigest() == vector_hash
for answer, digest in original_answers.items():
    assert hashlib.sha256(answer.read_bytes()).hexdigest() == digest
for side, rungs in backfilled['sides'].items():
    assert len(rungs) == 4
    for rung in rungs:
        if not rung.get('bridge', False):
            assert rung in full['sides'][side]
        for key in ('answers', 'control_answers'):
            assert len((root / rung[key]).read_text().splitlines()) == 20
print('BACKFILL_SMOKE_PASS real tiny model; four lower doses plus controls; original vector/answers/rungs/state unchanged; archive exact; cumulative timing preserved')
