"""End-to-end check of svdkv_resid's in-extraction calibration on a small hybrid model (GPU): constants, then C sweep.

    uv run --extra benchmark python scripts/scratch/svdkv_calib_check.py Qwen/Qwen3.5-0.8B
"""
import sys

import torch
from steering_lite import Vector
from steering_lite.config import _CONFIG_REGISTRY as C
from steering_lite.data import make_persona_pairs
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts/bsbench")
from walk import PERSONA_TEMPLATE, PERSONAS, resolve_layers  # noqa: E402

m = sys.argv[1]
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map="cuda").eval()
pos, neg = make_persona_pairs(tok, n_pairs=64, thinking=True, persona_pairs=PERSONAS, template=PERSONA_TEMPLATE, seed=0)
v = Vector.train(model, tok, pos, neg, C["svdkv_resid"](layers=resolve_layers(model, "svdkv_resid", None), dtype=torch.bfloat16),
                 batch_size=8, max_length=384)
print("CONSTANTS nu_scale", v.cfg.nu_scale, "r_scale", v.cfg.r_scale, "layers", v.cfg.layers)
qs = ["What is the capital of France? Answer in 2 short sentences.",
      "How do I compute the moment of inertia of my monolith codebase before splitting it into microservices? Answer in 2 short sentences."]
texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in qs]
tok.padding_side = "left"
b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
gen = lambda: tok.batch_decode(model.generate(**b, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)[:, b.input_ids.shape[1]:], skip_special_tokens=True)
print("bare", gen())
c0 = v.calibrate(model, tok).cfg.coeff
print("C0 of svdkv_resid", c0)
for m_ in (0.5, 1.0, 2.0):
    for sign in (1, -1):
        with v(model, C=sign * m_ * c0):
            print(f"C={sign * m_:+.1f}·C0", gen())
