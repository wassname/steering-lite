"""Rearrange existing README text; preserve historical wording separately. PI/OpenAI."""
import re
from pathlib import Path

path = Path('README.md')
text = path.read_text()
historical = re.findall(r'<details>\n<summary>Earlier .*?</details>', text, flags=re.S)
assert len(historical) == 2
archive = Path('slop/research/20261007_historical_bsbench_results/README.md')
assert not archive.exists()
archive.parent.mkdir(parents=True, exist_ok=True)
body = '\n\n'.join(historical)
body = re.sub(r'(?<=\]\()((?:assets|src|scripts|slop)/[^)]+|RESEARCH_JOURNAL\.md)(?=\))', r'../../../\1', body)
archive.write_text('# Historical BullshitBench README sections\n\nMoved without rewriting from README at fb1fd14. These use earlier setups and scores; see the root README for the current result. Rearrangement: PI/OpenAI.\n\n' + body + '\n')
for block in historical:
    text = text.replace(block + '\n\n', '', 1)
start = text.index('## Quickstart')
result = text.index('## Results')
run = text.index('To run the benchmark you need')
methods = text.index('## Methods and debugging')
header = text[:start].replace('[Try it](#quickstart) · [Results](#results)', '[Results](#results) · [Try it](#quickstart)')
quickstart = text[start:result]
main = text[result:run]
main += '[Earlier evaluations and exploratory prose](slop/research/20261007_historical_bsbench_results/README.md) are historical snapshots with different scoring.\n\n'
commands = '''## Run the benchmark

Use uv, just, pnpm, a Modal account and `OPENROUTER_API_KEY` in `.env`. The maintained recipes default to the main 9B setup: nonsense-question extraction pairs, control questions, three seeds and lower-dose samples. They incur GPU/API costs. Cached completed work is reused.

```bash
just check                          # real tiny-model library and benchmark smoke
just smoke-bsbench cache_mean_diff  # smoke a particular method
just sweep cache_mean_diff          # only this method, 9B, three seeds, 100 questions
just pull                           # download Modal artifacts
just results                        # judge, render the main 9B page, update its README figure, run browser checks
```

`just sweep-random` runs the separate 20-direction reference. Other models remain available through `scripts/bsbench/run_modal.py`; benchmark a changed preset before its first sweep.

To add a method: register its config and implementation, export the config, add it to `tests/test_pipeline.py::METHODS`, and choose a plot color in `scripts/bsbench/results.py`. Run the real smoke before a paid sweep. Lower-dose generation, cached backfill, judging and seed-mean plotting use the shared pipeline; no scratch script is required.

## Where things are

- `src/steering_lite/variants/`: one method per file, with math and paper references.
- `scripts/bsbench/`: maintained pipeline, `walk.py` → `judge.py` → `results.py`; `web/` contains the interactive view and browser checks.
- `data/bsbench/`: fixed evaluation questions, control questions and extraction pairs.
- `assets/bsbench/`: current README figure. Older flat asset paths remain for historical evidence links.
- `outputs/bsbench/`: generated answers, vectors, certificates and reports; not committed.
- `slop/`: dated research evidence, reviews and one-off scripts; `RESEARCH_JOURNAL.md` records decisions and results.

<!-- PI/OpenAI: current commands and path map, 2026-10-07. -->

'''
path.write_text(header + main + quickstart + commands + text[methods:])
