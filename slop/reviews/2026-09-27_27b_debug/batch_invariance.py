"""Bug check (PI/Claude, 2026-09-27): is extraction invariant to --extract-batch-size?
27B vectors were extracted with batch size 2, 4B with 8. Re-extract 4B vjp_delta, vjp_cache and mean_diff at bs=2
with walk.py's own extract_vector (same pairs, seed 0) into a temp dir; compare with the stored bs=8 vectors.
SHOULD: per-layer cos >= 0.99 for every method; ELSE padding/masking depends on batch and 27B vectors are suspect.
Run (GPU): cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-27_27b_debug/batch_invariance.py
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, ".")
import torch
from safetensors.torch import load_file
import walk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
stored = walk.model_dir(MODEL)
tmp = Path(tempfile.mkdtemp())
walk.model_dir = lambda model: tmp  # extract_vector caches under model_dir(...)/vectors; force a fresh extraction
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
for method in ("mean_diff", "vjp_delta", "vjp_cache"):
    args = walk.parse_args([method, "--seed", "0", "--cohort", "full", "--extract-batch-size", "2"])
    layers = walk.resolve_layers(model, method, args.layers)
    walk.extract_vector(args, model, tok, layers)
    new, old = load_file(str(tmp / "vectors" / f"{method}_s0.safetensors")), load_file(str(stored / "vectors" / f"{method}_s0.safetensors"))
    cs = {k: float(torch.nn.functional.cosine_similarity(new[k].float().flatten(), old[k].float().flatten(), dim=0)) for k in old if k.startswith("stacked.")}
    print(f"{method}: bs2 vs bs8 per-tensor cos min {min(cs.values()):+.4f} mean {sum(cs.values()) / len(cs):+.4f} (n={len(cs)})", flush=True)
