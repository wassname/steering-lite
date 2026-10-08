"""Run the real file judge on forward, reversed and identical answer pairs. PI/gpt-6.1-sol."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts/bsbench'))
import judge
from data import load_cohort

pilot = [json.loads(line) for line in (ROOT / 'slop/reviews/2026-10-05_judge_v4/pilot.jsonl').open()]
row = next(row for row in pilot if row['condition'] == 'vjp +C 0.315' and row['bare'] != row['steered'])
rows = [dict(row, condition='forward'), dict(row, condition='reversed', bare=row['steered'], steered=row['bare']),
        dict(row, condition='identical', steered=row['bare'])]
inputs = OUT / 'cli_input.jsonl'
inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows))
judge.CACHE = OUT / 'cli_cache.jsonl'
sys.argv = [str(ROOT / 'scripts/bsbench/judge_file.py'), str(inputs)]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    runpy.run_path(sys.argv[0], run_name='__main__')
print(output.getvalue(), end='')
parsed = {line.split('\t')[0]: dict(field.split('=', 1) for field in line.split('\t')[1:])
          for line in output.getvalue().splitlines()}
assert float(parsed['forward']['effect']) == -float(parsed['reversed']['effect'])
assert parsed['forward']['off_axis'] == parsed['reversed']['off_axis']
assert float(parsed['identical']['effect']) == 0.0
cohort = load_cohort()[row['scenario']]
args = cohort['prompt'], cohort['nonsensical_element']
have = judge.cached()
ab = have[judge.key(judge.pair_request(*args, row['bare'], row['steered']))]
ba = have[judge.key(judge.pair_request(*args, row['steered'], row['bare']))]
effect, off = judge.pair_change(ab, ba)
assert parsed['forward']['effect'] == f'{effect:+.2f}'
assert parsed['forward']['off_axis'] == f'{off:.2f}'
assert len(have) == 5, len(have)
print('FILE_JUDGE_PASS real API; both orders; identical=0; 5 deduplicated cache cells; matches judge.pair_change')
