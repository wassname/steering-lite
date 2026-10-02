"""Compare new measured support with the retained original population. — PI/OpenAI"""
import json
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/bsbench"))
from results import bootstrap, choose, curves_for, directed, frontier, method_curve, pareto_score, random_zones

site = json.loads(Path("outputs/bsbench/results/prompt-dev/points.json").read_text())
points = site["points"]
assert site["random_seeds"] == list(range(32))
old_zones = random_zones([p for p in points if p["method"] == "random" and p["seed"] < 11])
new_zones = random_zones(points)
assert json.loads(json.dumps(new_zones)) == site["zones"]
comparisons = []
for old, new in zip(old_zones, new_zones, strict=True):
    by_dose = dict(zip(old["doses"], old["bounds"], strict=True))
    diffs = [(C, abs(bound[2] - by_dose[C][2]), abs(bound[3] - by_dose[C][3]))
             for C, bound in zip(new["doses"], new["bounds"], strict=True) if C in by_dose and C is not None]
    comparisons.append({"percentile": new["percentile"], "shared_doses": len(diffs),
                        "max_bound_change": max(max(lo, hi) for _, lo, hi in diffs), "differences": diffs})
short = [p for p in points if p["method"] == "prompting_scale"]
assert len(short) == 62
assert all(not p["breakdown_reasons"] and not p["post_boundary"] for p in short)
controls = []
selected = next(r for r in site["summary"] if r["method"] == "prompting_scale")
for side, best in selected["best"].items():
    same = next(p for p in short if p["side"] == side and p["C"] == best["C"])
    opposite = next(p for p in short if p["side"] != side and p["C"] == best["C"])
    zero = next(p for p in short if p["side"] == side and p["C"] == 0)
    controls.append({"side": side, "gain": best["C"], "selected_effect": same["effect"],
                     "opposite_persona_effect": opposite["effect"], "same_persona_zero_effect": zero["effect"]})
raw_two = [p for p in points if p["method"] == "random" and p["C"] == 2]
assert len(raw_two) == 64
raw_diagnostic = {"dose": 2, "all_interventions": len(raw_two), "negative_effects": sum(p["effect"] < 0 for p in raw_two),
                  "positive_effects": sum(p["effect"] > 0 for p in raw_two), "mean_effect": mean(p["effect"] for p in raw_two),
                  "excluded": [{k: p[k] for k in ("seed", "side", "effect", "steered_damage", "breakdown_reasons", "post_boundary")}
                               for p in raw_two if not p["admissible"]]}
supports = {side: frontier(method_curve(points, "prompting_scale", side), include_endpoint=False) for side in ("+C", "-C")}
per_seed = []
for seed in range(32):
    subset = [p for p in points if p["method"] == "random" and p["seed"] == seed]
    curves = curves_for(subset, "random")
    score, best = pareto_score(curves)
    per_seed.append({"seed": seed, "score": score,
                     "negative_side_score": directed(best["-C"]) - best["-C"]["off_axis"]})
print("PER_SEED_RANDOM_NULL", sorted(per_seed, key=lambda p: p["negative_side_score"]))
report = {"signer": "PI/OpenAI", "per_seed_random_null": per_seed, "random_11_vs_32": comparisons,
          "old_zones": old_zones, "new_zones": new_zones,
          "random_at_two_unfiltered": raw_diagnostic,
          "selected_controls": controls,
          "short_gains": [{k: p[k] for k in ("C", "side", "effect", "off_axis", "steered_damage", "admissible")} for p in short],
          "pareto_supports": {side: [{k: p[k] for k in ("C", "effect", "off_axis")} for p in curve] for side, curve in supports.items()}}
Path("slop/reviews/2026-10-02_prompt_refinement/comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print("RANDOM_COMPARISON", comparisons)
print("RANDOM_UNFILTERED_C2", raw_diagnostic)
print("SELECTED_CONTROLS", controls)
for side in ("+C", "-C"):
    print("PASSING_GAINS", side, [p["C"] for p in short if p["side"] == side and p["admissible"]])
    print("EXCLUDED_GAINS", side, [p["C"] for p in short if p["side"] == side and not p["admissible"]])
for name in ("full", "27b-full", "olmo-full"):
    old_path = Path(f".local/prompt-refinement-before/{name}.json")
    new_path = Path(f"outputs/bsbench/results/{name}/points.json")
    before, after = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    assert before["blind"] == after["blind"]
    ci_changes = []
    for old_row, new_row in zip(before["summary"], after["summary"], strict=True):
        assert {k: v for k, v in old_row.items() if k not in ("ci", "ci_room")} == {k: v for k, v in new_row.items() if k not in ("ci", "ci_room")}
        if old_row["ci"] != new_row["ci"] or old_row["ci_room"] != new_row["ci_room"]:
            ci_changes.append({"method": new_row["method"], "before": old_row["ci"], "after": new_row["ci"],
                               "room_before": old_row["ci_room"], "room_after": new_row["ci_room"]})
    if ci_changes:
        choices = choose(after["points"])
        legacy_order = sorted(choices, key=lambda m: (m == "random", m.replace("vjp_resid", "vjp_delta").replace("vjp_value", "vjp_cache")))
        rng = random.Random(0)
        scenarios = [q["scenario"] for q in after["questions"]]
        before_rows = {r["method"]: r for r in before["summary"]}
        for method in legacy_order:
            curves = choices[method][0]
            if before_rows[method]["score"] is None:
                continue
            seeds = sorted({p["seed"] for p in after["points"] if p["method"] == method})
            low, high, _, room_low, room_high = bootstrap(curves, scenarios, seeds, rng)
            assert [low, high] == before_rows[method]["ci"], (name, method, "historical bootstrap replay")
            assert [room_low, room_high] == before_rows[method]["ci_room"]
        print("HISTORICAL_BOOTSTRAP_REPLAY_PASS", name, "old CI endpoints reproduced exactly using pre-rename ordering", ci_changes)
    extra_blind = 0
    identity = lambda p: (p["method"], p["seed"], p["side"], p["C"])
    old_points = {identity(p): p for p in before["points"]}
    new_points = {identity(p): p for p in after["points"]}
    assert old_points.keys() == new_points.keys()
    for old_point, new_point in [(old_points[k], new_points[k]) for k in old_points]:
        assert {k: v for k, v in old_point.items() if k != "questions"} == {k: v for k, v in new_point.items() if k != "questions"}
        for old_q, new_q in zip(old_point["questions"], new_point["questions"], strict=True):
            assert {k: v for k, v in old_q.items() if k != "blind"} == {k: v for k, v in new_q.items() if k != "blind"}
            assert old_q["blind"] is None or old_q["blind"] == new_q["blind"]
            extra_blind += old_q["blind"] is None and new_q["blind"] is not None
    print("ADDITIONAL_CACHED_BLIND_ATTACHMENTS", name, extra_blind)
    assert [z["bounds"] for z in before["zones"]] == [z["bounds"] for z in after["zones"]]
    before_seeds = sorted({p["seed"] for p in before["points"] if p["method"] == "random"})
    assert after["random_seeds"] == before_seeds
    assert after["plot_gap"] == 1
    print("FULL_SCIENTIFIC_REGRESSION_PASS", name, "keyed points, scores/selection, selected blind ratings, raw random bounds unchanged; CI changes", len(ci_changes), "; random seeds", before_seeds)
