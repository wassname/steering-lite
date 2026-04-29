"""Run bidirectional bench at iso-KL-calibrated coefficients.

Reads an iso-KL summary JSON and runs daily_dilemmas_benchmark.py at +/-c* per
method. Output dir matches `just bench-bidir` so `just si-flip` works.

Why: uncalibrated c=1.0 collapses spherical (pmass=0.013) and underdrives
cosine_gated (calibrated c*=28.6). Per-method iso-KL coeffs make SI comparable
across methods.

Usage:
  python scripts/bench_bidir_iso.py \
      --iso-kl outputs/iso_kl/iso_kl__Qwen--Qwen3-0.6B__...json \
      --out outputs/daily_dilemmas/v10_bidir_iso_qwen
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iso-kl", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-dtype", default="bfloat16")
    args = p.parse_args()

    iso = json.loads(args.iso_kl.read_text())
    model = iso["args"]["model"]
    layers = iso["args"]["layers"]  # 'mid' / 'all' / explicit
    args.out.mkdir(parents=True, exist_ok=True)

    # Baseline ref at coeff=0
    base_cmd = [
        "uv", "run", "--extra", "benchmark",
        "python", "scripts/daily_dilemmas_benchmark.py",
        "--model", model, "--method", "baseline", "--coeff", "0.0",
        "--layers", layers, "--device", args.device,
        "--torch-dtype", args.torch_dtype, "--output-dir", str(args.out),
    ]
    print(f">>> baseline c=0.0", flush=True)
    subprocess.run(base_cmd, check=True)

    for s in iso["summary"]:
        method = s["method"]
        c = float(s["calibrated_coeff"])
        for sign in (+1, -1):
            coeff = sign * c
            cmd = [
                "uv", "run", "--extra", "benchmark",
                "python", "scripts/daily_dilemmas_benchmark.py",
                "--model", model, "--method", method, "--coeff", f"{coeff:.6f}",
                "--layers", layers, "--device", args.device,
                "--torch-dtype", args.torch_dtype, "--output-dir", str(args.out),
            ]
            print(f">>> {method} c={coeff:+.4f} (calibrated |c*|={c:.4f})", flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
