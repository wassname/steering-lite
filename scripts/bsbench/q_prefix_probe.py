"""First check for q_prefix: does adding C·q̂* move attention onto the + persona tokens for C > 0 and onto − for C < 0?

For each bias b and C, on the 20 dev prompts (prefill, last token), mean over steered layers and heads of:
    mass  = attention on the whole persona prefix;   plus = on the "sycophantic" sentence;   share+ = plus / mass
SHOULD: at C = 0 mass small (b large enough); share+ rises with C and falls for C < 0. ELSE q* does not point at the prefix.

    uv run --extra benchmark python scripts/bsbench/q_prefix_probe.py [--model Qwen/Qwen3-4B]
"""
import argparse
import json

import torch
from loguru import logger
from steering_lite import QPrefixC, Vector
from steering_lite.data import make_persona_pairs
from steering_lite.variants.attn_site import _ACTIVE_PREFIX
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from walk import COHORT, COHORTS, PERSONA_TEMPLATE, PERSONAS, generation_inputs

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--n-pairs", type=int, default=64)
args = p.parse_args()
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if args.device == "cuda" else torch.float32, device_map=args.device).eval()
L = len(model.model.layers)
pos, neg = make_persona_pairs(tok, n_pairs=args.n_pairs, thinking=True, persona_pairs=PERSONAS, template=PERSONA_TEMPLATE, seed=0)
vec = Vector.train(model, tok, pos, neg, QPrefixC(layers=tuple(range(1, L)), dtype=model.dtype), batch_size=4, max_length=384)
rows = [json.loads(line) for line in COHORT.open()][COHORTS["dev"]]
prompts = generation_inputs(tok, rows)
table = []
with torch.no_grad():
    for b in (0.0, 4.0, 8.0):
        for C in (-4.0, -1.0, 0.0, 1.0, 4.0):
            vec.cfg.bias = b
            mass, plus = [], []
            with vec(model, C=C):
                for text in prompts:
                    model(tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device))
                    m = [P["mass"] for P in _ACTIVE_PREFIX.values()]
                    mass.append(sum(x[0] for x in m) / len(m))
                    plus.append(sum(x[1] for x in m) / len(m))
            M, Pl = sum(mass) / len(mass), sum(plus) / len(plus)
            table.append({"b": b, "C": C, "mass (all prefix)": M, "mass on +": Pl, "share+": Pl / M})
md = tabulate(table, headers="keys", tablefmt="pipe", floatfmt=".4f")
logger.info("SHOULD: share+ rises with C (≈0.5 at C=0 if both sentences equally read); mass small at C=0 for the chosen b\n{}", md)
print(md)
