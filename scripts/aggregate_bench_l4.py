"""Aggregate daily_dilemmas v5 L=4 calibrated runs into a markdown table."""
import json, glob, sys
from pathlib import Path

OUT_DIR = Path("outputs/daily_dilemmas/v5_L4_calibrated")

rows = []
for f in sorted(OUT_DIR.glob("*.json")):
    d = json.load(open(f))
    cfg, s = d["config"], d["summary"]
    rows.append({
        "method": cfg["method"],
        "coeff": cfg["coeff"],
        "target_effect": s["target_effect"],
        "leakage_mean": s["leakage_mean"],
        "surgical_informedness": s["surgical_informedness"],
        "si_tv": s.get("si_tv"),
        "si_js": s.get("si_js"),
    })

# Order: baseline first, then by SI desc
rows.sort(key=lambda r: (r["method"] != "baseline", -r["surgical_informedness"]))

print(f"# Daily Dilemmas, target=honesty, model=Qwen3.5-0.8B, L=4, iso-KL-calibrated (KL_p95=1.0 nat)\n")
print("| method | coeff | target_effect | leakage_mean | surgical_informedness |")
print("|---|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['method']} | {r['coeff']:.3f} | {r['target_effect']:+.3f} | {r['leakage_mean']:+.3f} | {r['surgical_informedness']:+.3f} |")
