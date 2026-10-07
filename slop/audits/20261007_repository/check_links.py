"""Check local links in the current and relocated README. PI/OpenAI."""
import re
from pathlib import Path

for path in (Path('README.md'), Path('slop/research/20261007_historical_bsbench_results/README.md')):
    checked = 0
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        target = target.split('#', 1)[0]
        assert (path.parent / target).exists(), (path, target)
        checked += 1
    print(f'LINKS_PASS {path}: {checked} local file targets')
assert Path('assets/bsbench/qwen3.5-9b.png').read_bytes() == Path('outputs/bsbench/results/v5-9b-3seeds/plot_readme.png').read_bytes()
print('FIGURE_PASS README asset equals the generated main 9B plot')
