"""Paired full-score comparison using production dose selection and resampling. PI/OpenAI."""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bsbench"))
from results import N_BOOT, curves_for, pareto_score, resample

OUT = Path(__file__).parent
site = json.loads((ROOT / "outputs/bsbench/results/full/points.json").read_text())
points = [p for p in site["points"] if p["seed"] in (range(11) if p["method"] == "random" else range(3))]
published_scores = {"mean_diff": 0.37353333333333333, "corda_pca": 0.26213333333333333, "random": -0.06673636363636365}
for method, expected in published_scores.items():
    actual = pareto_score(curves_for(points, method))[0]
    assert math.isclose(actual, expected, abs_tol=1e-12), (method, actual, expected)
    print(f"Published baseline restored: {method} = {actual:.8f}", flush=True)

methods = ("sink_split_resid", "sink_split", "mean_diff", "random")
curves = {m: curves_for(points, m) for m in methods}
scores = {m: pareto_score(curves[m])[0] for m in methods}
scenarios = [q["scenario"] for q in site["questions"]]
assert len(scenarios) == 100
assert all({p["seed"] for p in points if p["method"] == m} == {0, 1, 2} for m in methods if m != "random")
rng = random.Random(0)
null_rng = random.Random(1)
pairs = (("sink_split_resid", "mean_diff"), ("sink_split", "mean_diff"), ("sink_split_resid", "sink_split"), ("sink_split", "random"))
diffs = {pair: [] for pair in pairs}
for i in range(N_BOOT):
    seeds = rng.choices([0, 1, 2], k=3)
    questions = rng.choices(scenarios, k=100)
    null_seeds = null_rng.choices(list(range(11)), k=11)
    drawn = {m: pareto_score({side: resample(c, questions, null_seeds if m == "random" else seeds) for side, c in curves[m].items()})[0] for m in methods}
    assert all(math.isfinite(v) for v in drawn.values()), "undefined paired difference: no admissible dose in a draw"
    for a, b in pairs:
        diffs[a, b].append(drawn[a] - drawn[b])
    if i % 100 == 0:
        print(f"paired bootstrap {i}/{N_BOOT}", flush=True)

rows = ["# Full sink-split comparison", "", "PI/OpenAI. Qwen3.5-4B, 100 questions, extraction seeds 0-2. Production resample and pareto_score; 1000 paired hierarchical draws with shared questions and dose selection redone. Learned methods share the seed draws; random draws its 11 seeds independently. Intervals remain conditional on the original full-data admissibility decisions.", "", "| comparison | score difference | 90% paired interval |", "|---|---:|---:|"]
report = []
for a, b in pairs:
    d = sorted(diffs[a, b]); low, high = d[int(.05 * len(d))], d[int(.95 * len(d)) - 1]
    report.append({"a": a, "b": b, "difference": scores[a] - scores[b], "ci90": [low, high]})
    rows.append(f"| {a} - {b} | {scores[a] - scores[b]:+.3f} | [{low:+.3f}, {high:+.3f}] |")
rows += ["", "Full contains the 20 dev questions. Doses are selected on these same questions. This comparison does not identify the causal contribution of the attention component or validate on an independent held-out benchmark.", ""]
(OUT / "full-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
(OUT / "full-comparison.md").write_text("\n".join(rows))

model_dir = ROOT / "outputs/bsbench" / site["model_dir"]
def answers(path):
    return {r["scenario"]: r for r in map(json.loads, path.open())}
bare = answers(model_dir / "answers/bare/bare.jsonl")
selected = {m: pareto_score(curves[m])[1] for m in methods}
texts = {}
for method in methods:
    for side in ("-C", "+C"):
        dose = selected[method][side]["C"]
        texts[method, side] = answers(model_dir / "answers" / f"{method}_s0" / f"{side}_C{dose:.10g}.jsonl")
examples = ["# Complete answer samples", "", "PI/OpenAI. Fixed selection: dataset indices 1,11,...91, all outside the every-fifth dev subset. Seed 0 at each method's score-selected dose (three learned seeds, eleven random seeds); all text is verbatim. Includes both steering signs and bare.", ""]
for q in site["questions"][1::10]:
    s = q["scenario"]
    examples.extend([f"## {s}", "", q["prompt"], "", "### Bare", "", bare[s]["text"], ""])
    for method in methods:
        for side in ("-C", "+C"):
            examples.extend([f"### {method} {side}, C={selected[method][side]['C']:.10g}", "", texts[method, side][s]["text"], ""])
(OUT / "full-examples.md").write_text("\n".join(examples))
health = ["", "## Coverage and breakdown", "", "Every recorded answer file was checked against the exact full set of 100 scenarios, with no duplicate rows. Sources: `outputs/bsbench/" + site["model_dir"] + "/walks/sink_split{,_resid}_s{0,1,2}_full.json` and their answer paths. Rows below show the dose before the first health failure, the first failure, and the following dose. Counts have denominator 100; KL is the logged RMS token KL, in nats.", "", "| method | seed | side | dose C | KL | unfinished | role leaks | repeated | health failure |", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
walk_seconds = 0.0
answer_count = 0
for method in ("sink_split", "sink_split_resid"):
    for seed in range(3):
        path = model_dir / "walks" / f"{method}_s{seed}_full.json"
        cert = json.loads(path.read_text())
        assert cert["status"] == "COMPLETE"
        walk_seconds += cert["timing"]["total_s"]
        for r in cert["rungs"]:
            for side in ("-C", "+C"):
                answers_at = list(map(json.loads, (model_dir / r[side]["answers"]).open()))
                assert len(answers_at) == 100 and {a["scenario"] for a in answers_at} == set(scenarios)
                answer_count += len(answers_at)
        for side in ("-C", "+C"):
            assert cert["state"][side]["boundary"] is not None
            bad = next(i for i, r in enumerate(cert["rungs"]) if r[side]["breakdown_reasons"])
            assert 0 < bad < len(cert["rungs"]) - 1
            for r in cert["rungs"][bad-1:bad+2]:
                h = r[side]["stats"]
                reason = ", ".join(r[side]["breakdown_reasons"]) or "none"
                health.append(f"| {method} | {seed} | {side} | {r['coefficient']:.3g} | {r['kl_rms'][side]:.3f} | {h['unfinished']} | {h['role_leaks']} | {h['repeated']} | {reason} |")
health.extend(["", f"Coverage passed: six completed walks, {answer_count} answer rows. Measured sum of walk times: {walk_seconds:.1f} GPU seconds on L40S. GPU-only estimate at the previously recorded $0.000542/s list rate: ${walk_seconds * .000542:.2f}; excludes CPU/memory charges and is not an invoice.", "", "The pre-rename full scores used for the restoration assertions were copied from `.local/method-naming/full-before.json`, the report snapshot preserved by the method-name migration (05a83da). They are constants in this script so rerunning does not need that private backup.", ""])
with (OUT / "full-comparison.md").open("a") as f:
    f.write("\n".join(health))
print("\n".join(rows + health), flush=True)
