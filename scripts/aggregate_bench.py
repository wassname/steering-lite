"""Aggregate daily_dilemmas v6 L=7..19 calibrated runs.

Includes per-value baseline + steered absolute scores, deltas, and SI.
"""
import json, glob, sys
from pathlib import Path

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/daily_dilemmas/v6_L7-19_calibrated")

# Find baseline
files = sorted(OUT_DIR.glob("*.json"))
baseline = None
for f in files:
    d = json.load(open(f))
    if d["config"]["method"] == "baseline":
        baseline = d
        break

if baseline is None:
    print("No baseline run found", file=sys.stderr)
    sys.exit(1)

target = baseline["config"]["target"]
b_eff = baseline["effects"]
base_target_score = b_eff[target]["base"]  # absolute log-prob on target-tagged actions

rows = []
for f in files:
    d = json.load(open(f))
    cfg, s = d["config"], d["summary"]
    if cfg["method"] == "baseline":
        continue
    rows.append({
        "method": cfg["method"],
        "coeff": cfg["coeff"],
        "target_steered": d["effects"][target]["steered"],
        "target_effect": s["target_effect"],
        "leakage_mean": s["leakage_mean"],
        "surgical_informedness": s["surgical_informedness"],
    })

rows.sort(key=lambda r: -r["surgical_informedness"])

layers = baseline["config"]["layers"]
n_layers = len(layers)
total_layers = 24  # Qwen3.5-0.8B
pct = f"{layers[0]/total_layers:.0%}-{(layers[-1]+1)/total_layers:.0%}"

print(f"# Daily Dilemmas, target={target}, model={baseline['config']['model']}")
print(f"L={layers[0]}..{layers[-1]} ({n_layers}/{total_layers} layers, {pct} depth), seed=0, n_train=n_eval={baseline['config']['n_train']}")
print(f"Coeffs iso-KL-calibrated to KL_p95=1.0 nat (greedy, T=20, N_calib=4).\n")
print(f"baseline target log-prob = {base_target_score:+.3f}\n")

print("| method | coeff | target_lp_steered | target_effect Δ | leakage_mean Δ | surgical_informedness |")
print("|---|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['method']} | {r['coeff']:.3f} | {r['target_steered']:+.3f} | {r['target_effect']:+.3f} | {r['leakage_mean']:+.3f} | {r['surgical_informedness']:+.3f} |")

print("\nΔ = steered - baseline. SI = target_effect - leakage_mean (higher = more selective at matched KL).")
