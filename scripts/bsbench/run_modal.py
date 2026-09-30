"""Fan the dose walks out over Modal GPUs, one container per (method, seed).

Adapted from vjp-steering 7f0782a `scripts/run_modal.py`. Changes: pip pins instead of uv_sync,
flash-linear-attention in the image (Qwen3.5's linear-attention layers otherwise fall back to
slow torch code), steering-lite methods and a dev/full cohort, and outputs/bsbench on the Volume.

    modal run scripts/bsbench/run_modal.py::smoke
    modal run scripts/bsbench/run_modal.py --cohort dev --methods mean_diff,pca --seeds 0
    modal volume get --force steering-lite-bsbench-v3 bsbench outputs/
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")  # container copy is /root/run_modal.py
MODEL = "Qwen/Qwen3.5-4B"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "torch==2.11.0", "transformers==5.12.1", "accelerate==1.13.0", "safetensors==0.7.0",
        "einops==0.8.2", "jaxtyping==0.3.9", "beartype==0.22.9", "loguru==0.7.3", "tabulate==0.10.0",
        "tqdm==4.67.3", "numpy==2.4.4", "flash-linear-attention==0.5.2", "fla-core==0.5.2",
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "PYTHONPATH": "/repo/src"})
    .add_local_dir(REPO / "src", "/repo/src")
    .add_local_dir(REPO / "scripts", "/repo/scripts")
    .add_local_dir(REPO / "data", "/repo/data")
)
app = modal.App("steering-lite-bsbench-v3", image=image)
cache = modal.Volume.from_name("steering-lite-bsbench-v3", create_if_missing=True)


@app.function(gpu=os.environ.get("BSBENCH_GPU", "L40S"), volumes={"/cache": cache}, timeout=6 * 60 * 60)
def run(argv: list[str]) -> str:
    """One walk of scripts/bsbench/walk.py; outputs/bsbench and outputs/bsbench-smoke live on the Volume."""
    from huggingface_hub import snapshot_download

    Path("/cache/bsbench").mkdir(parents=True, exist_ok=True)
    Path("/cache/bsbench-smoke").mkdir(parents=True, exist_ok=True)
    Path("/repo/outputs").mkdir(exist_ok=True)
    for name in ("bsbench", "bsbench-smoke"):
        if not Path(f"/repo/outputs/{name}").exists():
            os.symlink(f"/cache/{name}", f"/repo/outputs/{name}")
    snapshot_download(argv[argv.index("--model") + 1] if "--model" in argv else MODEL)
    cache.commit()
    try:
        subprocess.run([sys.executable, "scripts/bsbench/walk.py", *argv], cwd="/repo", check=True)
    finally:
        cache.commit()
    return " ".join(argv)


def cached_on_volume(argv: list[str]) -> bool:
    """Read the walk certificate from the Volume before spawning, so a finished walk starts no GPU container."""
    sys.path.insert(0, str(REPO / "scripts/bsbench"))
    import walk  # local import: needs the repo venv (torch, transformers)

    args = walk.parse_args(argv)
    if args.smoke or args.probe:
        return False
    if args.profile or args.vjp_check or args.vjp_split:
        try:
            b"".join(cache.read_file(str(walk.model_dir(args.model).relative_to(walk.OUT.parent) / walk.mode_output(args))))
            return True
        except FileNotFoundError:
            return False
    path = walk.model_dir(args.model).relative_to(walk.OUT.parent) / "walks" / f"{args.name}_s{args.seed}_{args.cohort}.json"
    try:
        certificate = json.loads(b"".join(cache.read_file(str(path))))
    except FileNotFoundError:
        return False
    return walk.walk_done(certificate, args)


@app.local_entrypoint()
def main(methods: str = "mean_diff,pca,vjp_resid", seeds: str = "0", cohort: str = "dev", extra: str = ""):
    jobs = [(method, seed) for seed in seeds.split(",") for method in methods.split(",")]
    argvs = {job: [job[0], "--seed", job[1], "--cohort", cohort, *extra.split()] for job in jobs}
    todo = [job for job in jobs if not cached_on_volume(argvs[job])]
    for method, seed in sorted(set(jobs) - set(todo)):
        print(f"WALK_CACHED_LOCAL\t{method}\ts{seed}\tcohort={cohort} (certificate COMPLETE on the Volume; no container started)")
    handles = {job: run.spawn(argvs[job]) for job in todo}
    failed = []
    for (method, seed), handle in handles.items():
        try:
            print(f"DONE\t{method}\ts{seed}\t{handle.get()}")
        except Exception as error:  # collect, so one dead walk does not hide the others; raise at the end
            print(f"FAILED\t{method}\ts{seed}\t{error!r}")
            failed.append(f"{method} s{seed}")
    if failed:
        raise SystemExit(f"{len(failed)} of {len(handles)} walks FAILED: {', '.join(failed)}")


@app.local_entrypoint()
def smoke():
    """Same image, mounts and Volume as the real fan-out, real Qwen3.5-4B, 8-token answers, 2 rungs."""
    print(run.remote("vjp_value --seed 0 --cohort dev --smoke --n-pairs 8 --max-rungs 2".split()))
    print(run.remote("mean_diff --seed 0 --cohort dev --smoke --n-pairs 8 --max-rungs 2".split()))
