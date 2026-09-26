"""Is a saved VJP-family vector mostly the target-layer contrast c carried by the residual identity path?

J_l.T c = c + (attention/MLP pullback), because h_target = h_l + sum of later block writes. vjp_delta cancels
the c term (positive minus negative); mean_vjp and wiki_mean_vjp keep it. This prints, per source layer,
the share of each unit vector along c/|c| (cosine; same residual basis, so the comparison is meaningful)
and along the mean_diff vector at that layer. Reviewer finding #1, 2026-09-26.

    uv run --extra benchmark modal run scripts/bsbench/diag_identity_path.py
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), "/repo/scripts/bsbench"]
import modal
from run_modal import cache, image

app = modal.App("bsbench-diag-identity", image=image)
METHODS = ("vjp_delta", "mean_vjp", "wiki_mean_vjp", "mean_diff", "random")


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=1800)
def remote() -> None:
    Path("/repo/outputs").mkdir(exist_ok=True)
    if not Path("/repo/outputs/bsbench").exists():
        os.symlink("/cache/bsbench", "/repo/outputs/bsbench")
    subprocess.run([sys.executable, "scripts/bsbench/diag_identity_path.py", "--compute"], cwd="/repo", check=True)


@app.local_entrypoint()
def main() -> None:
    remote.remote()


def compute() -> None:
    import torch
    from steering_lite import Vector
    from steering_lite.variants.vjp_delta import _target_mean
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import walk

    args = walk.parse_args(["vjp_delta"])
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    pos, neg = walk.make_persona_pairs(tok, n_pairs=args.n_pairs, thinking=True, persona_pairs=walk.PERSONAS,
                                       template=walk.PERSONA_TEMPLATE, seed=0)
    vectors = {m: Vector.load(str(walk.model_dir(args.model) / "vectors" / f"{m}_s0.safetensors")) for m in METHODS}
    target = vectors["mean_vjp"].cfg.target_layer
    c = _target_mean(model, tok, pos, target, args.extract_batch_size, args.max_length) - _target_mean(
        model, tok, neg, target, args.extract_batch_size, args.max_length)
    c_hat = (c / c.norm()).cpu()
    print(f"target layer {target}; SHOULD: vjp_delta near 0 on cos(v,c) (c cancels); mean_vjp/wiki near 1 means they are mostly c")
    print("layer | " + " | ".join(f"cos({m},c)" for m in METHODS) + " | " + " | ".join(f"cos({m},mean_diff)" for m in METHODS[:3]))
    for layer in sorted(vectors["vjp_delta"].stacked):
        v = {m: vectors[m].stacked[layer]["v"].float().sum(0) for m in METHODS}
        v = {m: x / x.norm() for m, x in v.items()}
        row = [f"{(v[m] @ c_hat).item():+.3f}" for m in METHODS] + [f"{(v[m] @ v['mean_diff']).item():+.3f}" for m in METHODS[:3]]
        print(f"{layer} | " + " | ".join(row))


if __name__ == "__main__" and "--compute" in sys.argv:
    compute()
