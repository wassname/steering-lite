"""Compare calibration coeffs across two iso JSON sources (e.g. free-dNLL vs greedy-KL).

Usage:
  uv run python scripts/compare_iso_coeffs.py \\
    --a outputs/iso_tv/iso__...free_dnll0.1...json --label-a "free-dNLL=0.10" \\
    --b outputs/iso_kl/iso_kl__...greedy_kl_p95_1.0...json --label-b "greedy KL_p95=1.0" \\
    --out docs/compare_iso_coeffs.md
"""
import argparse
import json
from pathlib import Path


def load_summary(path):
    iso = json.load(open(path))
    return {(r["seed"], r["method"]): r for r in iso["summary"]}, iso.get("args", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, args_a = load_summary(args.a)
    B, args_b = load_summary(args.b)
    keys = sorted(set(A.keys()) & set(B.keys()))

    md = []
    md.append(f"# Calibration coefficients: {args.label_a} vs {args.label_b}\n")
    md.append(f"- model: `{args_a.get('model')}` (must match: `{args_b.get('model')}`)")
    md.append(f"- A: `{Path(args.a).name}` (target={args_a.get('target_metric_value', args_a.get('target_kl', '?'))})")
    md.append(f"- B: `{Path(args.b).name}` (target={args_b.get('target_kl', '?')} stat={args_b.get('target_stat', '?')})")
    md.append("")
    md.append("| seed | method | coeff_A | coeff_B | ratio B/A | A_metric | B_calib_p95 | B_val_samp_p95 |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for (seed, method) in keys:
        ra, rb = A[(seed, method)], B[(seed, method)]
        ca, cb = ra["calibrated_coeff"], rb["calibrated_coeff"]
        if ca is None or cb is None:
            continue
        # A's primary metric varies; show whatever is there
        a_metric = ra.get("metric_value", ra.get("calib_kl_mean", "?"))
        b_p95 = rb.get("calib_kl_p95", "?")
        b_val_p95 = rb.get("val_sampled_kl_p95", "?")
        a_metric_s = f"{a_metric:.3f}" if isinstance(a_metric, (int, float)) else a_metric
        b_p95_s = f"{b_p95:.3f}" if isinstance(b_p95, (int, float)) else b_p95
        b_val_p95_s = f"{b_val_p95:.3f}" if isinstance(b_val_p95, (int, float)) else b_val_p95
        md.append(f"| {seed} | {method} | {ca:.4f} | {cb:.4f} | {cb/ca:.2f} | {a_metric_s} | {b_p95_s} | {b_val_p95_s} |")

    md.append("")
    md.append("## How to read")
    md.append("- coeff_A and coeff_B are the per-method scalar that the steering hook multiplies the direction by.")
    md.append("- ratio B/A < 1 means the new metric needs a SMALLER coeff to satisfy its constraint (the new constraint is stricter for that method).")
    md.append("- B_calib_p95 = the 95th percentile of per-token KL(steer || base) under GREEDY decoding, over the first 20 generated tokens, on 4 calibration prompts. Bisected to ≈ 1.0 nat by construction.")
    md.append("- B_val_samp_p95 = same statistic but on 8 held-out prompts under SAMPLED decoding (temp=1, top_p=1, top_k=20). This is what you'll see at deployment.")

    out = Path(args.out) if args.out else Path("docs/compare_iso_coeffs.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
