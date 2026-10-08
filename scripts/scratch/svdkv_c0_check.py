"""svdkv at C = 0 vs detached: next-token KL (all positions and last token) and max |Δlogit|, 8 chat prompts from the dev cohort.

    uv run --extra benchmark python scripts/scratch/svdkv_c0_check.py Qwen/Qwen3.5-0.8B 320
    uv run --extra benchmark python scripts/scratch/svdkv_c0_check.py Qwen/Qwen3.5-4B calib   # ν from the in-extraction iso-KL
"""
import json
import sys

import torch
import torch.nn.functional as F
from steering_lite import Vector
from steering_lite.config import _CONFIG_REGISTRY as C
from steering_lite.data import make_persona_pairs
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts/bsbench")
from walk import COHORT, PERSONA_TEMPLATE, PERSONAS, generation_inputs, resolve_layers  # noqa: E402

m, nu = sys.argv[1], (None if sys.argv[2] == "calib" else float(sys.argv[2]))
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev).eval()
pos, neg = make_persona_pairs(tok, n_pairs=32, thinking=True, persona_pairs=PERSONAS, template=PERSONA_TEMPLATE, seed=0)
v = Vector.train(model, tok, pos, neg, C["svdkv"](layers=resolve_layers(model, "svdkv", None), dtype=model.dtype, nu_scale=nu),
                 batch_size=4, max_length=384)
rows = [json.loads(line) for line in COHORT.open()][:100:12][:8]
kl_all, kl_last, dmax = [], [], []
with torch.no_grad():
    for text in generation_inputs(tok, rows):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        base = model(ids).logits.float().log_softmax(-1)[0]
        with v(model, C=0.0):
            steer = model(ids).logits.float().log_softmax(-1)[0]
        kl = F.kl_div(steer, base, log_target=True, reduction="none").sum(-1)  # KL(base || steered) per position
        kl_all.append(kl.mean().item()); kl_last.append(kl[-1].item()); dmax.append((steer - base).abs().max().item())
print(f"C0CHECK {m} nu={v.cfg.nu_scale:.4g} n={len(kl_all)}: KL mean over positions {sum(kl_all)/len(kl_all):.4f} nats (max prompt {max(kl_all):.4f}); "
      f"last-token KL mean {sum(kl_last)/len(kl_last):.4f} (max {max(kl_last):.4f}); max |Δ log-prob| {max(dmax):.3f}")
