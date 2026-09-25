"""Scratch: +C answers where Jev sees much more damage than DeepSeek (chars, spherical, linear_act at their +C score dose)."""
import sys; sys.path.insert(0, ".")
from judge import default_model_dir
from results import build_points, jev_points, method_curve, pareto_score
md = default_model_dir()
ds = build_points(md, "full", set()); jv = jev_points(ds, md)
idx = {(p["method"], p["seed"], p["C"], p["side"]): i for i, p in enumerate(ds)}
rows = []
for m in ("chars", "spherical", "linear_act"):
    best = pareto_score({"+C": method_curve(ds, m, "+C")})[1]["+C"]
    for s in (0, 1, 2):
        i = idx[(m, s, best["C"], "+C")]
        for qd, qj in zip(ds[i]["questions"], jv[i]["questions"]):
            rows.append((qj["jev_damage"] - qd["steered_off_axis"] * 4 / 5, m, s, best["C"], qd, qj))
rows.sort(key=lambda r: -r[0])
import statistics
print("mean steered damage at +C score dose: DeepSeek (0-5) %.2f, Jev (0-4) %.2f" % (statistics.mean(r[4]["steered_off_axis"] for r in rows), statistics.mean(r[5]["jev_damage"] for r in rows)))
for gap, m, s, C, qd, qj in rows[:4]:
    print(f"\n=== {m} s{s} +C C={C:.3g} {qd['scenario']}: DeepSeek steered off {qd['steered_off_axis']:.1f}/5, Jev damage {qj['jev_damage']:.2f}/4")
    print(qd["text"][:500].replace("\n", " "))
    print("  DeepSeek evidence:", qd["evidence"])
