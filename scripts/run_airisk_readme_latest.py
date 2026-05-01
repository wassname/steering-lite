"""Run fresh AIRiskDilemmas benches for README tables."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

logger.remove()
logger.add(lambda x: tqdm.write(x, end=""), level="INFO", colorize=False, format="{message}")


METHODS = [
    "mean_diff",
    "mean_centred",
    "pca",
    "topk_clusters",
    "cosine_gated",
    "sspace",
    "spherical",
    "directional_ablation",
    "chars",
    "linear_act",
    "angular_steering",
]

COEFFS = {
    "spherical": 0.1,
    "angular_steering": 0.1,
    "linear_act": 0.1,
}


def _run(cmd: list[str]) -> None:
    logger.info(">>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _bench_cmd(*, model: str, method: str, target: str, coeff: float, out: Path,
               layers: str, device: str, torch_dtype: str, n_train: int | None = None,
               n_eval: int | None = None) -> list[str]:
    cmd = [
        "uv", "run", "--extra", "benchmark",
        "python", "scripts/airisk_dilemmas_benchmark.py",
        "--model", model,
        "--method", method,
        "--target", target,
        "--coeff", str(coeff),
        "--layers", layers,
        "--device", device,
        "--torch-dtype", torch_dtype,
        "--output-dir", str(out),
    ]
    if n_train is not None:
        cmd.extend(["--n-train", str(n_train)])
    if n_eval is not None:
        cmd.extend(["--n-eval", str(n_eval)])
    return cmd


def _latest_json(out: Path) -> Path:
    files = [p for p in out.glob("*.json") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"no json outputs in {out}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _target_effect(out: Path) -> float:
    data = json.loads(_latest_json(out).read_text())
    return float(data["summary"]["target_effect"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--target", default="Truthfulness")
    ap.add_argument("--out", type=Path, default=Path("outputs/airisk_dilemmas/readme_latest"))
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--probe-n-train", type=int, default=8)
    ap.add_argument("--probe-n-eval", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    probe_out = args.out / "probe"
    full_out = args.out / "full"
    probe_out.mkdir(parents=True, exist_ok=True)
    full_out.mkdir(parents=True, exist_ok=True)

    _run(_bench_cmd(
        model=args.model, method="baseline", target=args.target, coeff=0.0,
        out=full_out, layers=args.layers, device=args.device, torch_dtype=args.torch_dtype,
    ))

    for method in tqdm(METHODS, desc="methods", mininterval=60):
        coeff = COEFFS.get(method, 2.0)
        plus_out = probe_out / f"{method}_plus"
        minus_out = probe_out / f"{method}_minus"
        plus_out.mkdir(parents=True, exist_ok=True)
        minus_out.mkdir(parents=True, exist_ok=True)

        _run(_bench_cmd(
            model=args.model, method=method, target=args.target, coeff=coeff,
            out=plus_out, layers=args.layers, device=args.device, torch_dtype=args.torch_dtype,
            n_train=args.probe_n_train, n_eval=args.probe_n_eval,
        ))
        plus_delta = _target_effect(plus_out)

        if plus_delta > 0:
            chosen_sign = +1.0
        else:
            _run(_bench_cmd(
                model=args.model, method=method, target=args.target, coeff=-coeff,
                out=minus_out, layers=args.layers, device=args.device, torch_dtype=args.torch_dtype,
                n_train=args.probe_n_train, n_eval=args.probe_n_eval,
            ))
            minus_delta = _target_effect(minus_out)
            chosen_sign = +1.0 if plus_delta >= minus_delta else -1.0

        final_out = full_out / method
        final_out.mkdir(parents=True, exist_ok=True)
        _run(_bench_cmd(
            model=args.model, method=method, target=args.target, coeff=chosen_sign * coeff,
            out=final_out, layers=args.layers, device=args.device, torch_dtype=args.torch_dtype,
        ))


if __name__ == "__main__":
    main()
