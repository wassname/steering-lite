"""q_slot on Qwen3-4B: per layer, ν = ‖v_sink‖ per KV head (the write cap) vs sink_value's per-head write C·‖v̂*_g‖ at its best
-C dose, and the net weight moved between the two sink halves at the walk's doses (20 dev prompts, last token).

    uv run --extra benchmark python q_slot_probe.py
"""
import json

import torch
from steering_lite import Vector
from steering_lite.variants.attn_site import _ACTIVE_SLOT
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from walk import COHORT, COHORTS, generation_inputs, model_dir

MODEL = "Qwen/Qwen3-4B"
D = model_dir(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
rows = [json.loads(line) for line in COHORT.open()][COHORTS["dev"]]
prompts = generation_inputs(tok, rows)
slot = Vector.load(str(D / "vectors/q_slot_s0.safetensors"))
sv = Vector.load(str(D / "vectors/sink_value_s0.safetensors"))
d = model.config.head_dim
table = []
for L in (3, 8, 15, 22, 29, 34):
    nu = slot.shared[L]["vsink"].norm(dim=-1)  # [KVH]
    sv_head = sv.stacked[L]["x"][0].view(-1, d).norm(dim=-1)  # per-head share of the unit layer vector
    table.append({"layer": L, "ν=‖v_sink‖ mean": nu.mean().item(), "sink_value write per head at C=12.7": 12.7 * sv_head.mean().item(),
                  "ratio": (12.7 * sv_head.mean() / nu.mean()).item()})
print(tabulate(table, headers="keys", tablefmt="pipe", floatfmt=".3f"))
moved = []
with torch.no_grad():
    for C in (0.0, 13.4, 26.8, -26.8, 53.6):
        vals = []
        with slot(model, C=C):
            for text in prompts:
                model(tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda())
                vals.append(sum(P["read"] for P in _ACTIVE_SLOT.values()) / len(_ACTIVE_SLOT))
        moved.append({"C": C, "net + half weight (mean over layers, heads)": sum(vals) / len(vals)})
print(tabulate(moved, headers="keys", tablefmt="pipe", floatfmt=".4f"))
