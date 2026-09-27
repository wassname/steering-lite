"""Per-rung audit of the dose walks, 4B vs 27B (PI/Claude, 2026-09-27): is the useful region sampled, where is breakdown?
Columns per side: KL (nats, rms), on-axis toward the side's target, off-axis, mean steered damage, admissible, breakdown reasons.
Run: python slop/reviews/2026-09-27_27b_debug/walk_audit.py > slop/reviews/2026-09-27_27b_debug/walk_audit.md
"""
import json, glob, sys
methods = sys.argv[1:] or ["vjp_delta", "vjp_cache", "mean_diff", "chars", "random"]
for model, res, pat in (("4B", "full", "Qwen--Qwen3.5-4B"), ("27B", "27b-full", "Qwen--Qwen3.5-27B"), ("OLMo-32B", "olmo-full", "allenai--OLMo-2-0325-32B-Instruct")):
    md = glob.glob(f"outputs/bsbench/{pat}-g*")[0]
    pts = {(p["method"], p["seed"], p["C"], p["side"]): p for p in json.load(open(f"outputs/bsbench/results/{res}/points.json"))["points"]}
    best = {r["method"]: r["best"] for r in json.load(open(f"outputs/bsbench/results/{res}/points.json"))["summary"]}
    for m in methods:
        c = json.load(open(f"{md}/walks/{m}_s0_full.json"))
        print(f"\n### {model} {m} s0: c0={c['c0']:.3g} stride={c['stride']} rungs={len(c['rungs'])} state={c['state']}")
        bC = {s: (best[m][s] or {}).get("C") for s in ("+C", "-C")}
        print("| C | grid | " + " | ".join(f"{s} KL | {s} on | {s} off | {s} dmg | {s} ok" for s in ("-C", "+C")) + " |")
        print("|" + "---|" * 12)
        for r in sorted(c["rungs"], key=lambda r: r["coefficient"]):
            C = r["coefficient"]; cells = []
            for s in ("-C", "+C"):
                p = pts.get((m, 0, C, s))
                kl = r["kl_rms"][s] if isinstance(r["kl_rms"], dict) else None
                if p is None:
                    cells.append(f"{kl:.2f} | — | — | — | —"); continue
                on = p["effect"] if s == "+C" else -p["effect"]
                ok = "Y" if p["admissible"] else ("N:" + ",".join(p["breakdown_reasons"]) if p["breakdown_reasons"] else ("N:post" if p["post_boundary"] else "N:dmg"))
                star = "★" if bC[s] is not None and abs(bC[s] - C) < 1e-9 else ""
                cells.append(f"{kl:.2f} | {on:+.2f}{star} | {p['off_axis']:.2f} | {p['steered_damage']:.2f} | {ok}")
            print(f"| {C:.3g} | {r['grid_index']} | " + " | ".join(cells) + " |")
