"""Where does sink_value's effect come from? Per layer, on the 20 BS-bench dev prompts plus their bare answers (teacher-forced):

    a_first[L, h]  = mean over query positions t ≥ 16 of A_t,first           (attention on the first token, the sink)
    a_punct[L, h]  = mean over t of Σ_{s punctuation, s ≠ first} A_t,s    (attention on full stops, newlines, commas...)
    write[L]       = mean_t ‖ Σ_h A_t,first · W_O^h v̂*_g(h) ‖                  (size of sink_value's residual write at C = 1)
    write_all[L]   = ‖ Σ_h W_O^h v̂*_g(h) ‖                                    (the same write if every head read only the sink)

Reads the saved sink_value seed-0 vector. Writes outputs/bsbench/results/q3-4b-dev/sink_probe.md.

    uv run --extra benchmark python scripts/bsbench/sink_probe.py
"""
import json
from pathlib import Path

import torch
from loguru import logger
from steering_lite import Vector
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from walk import COHORT, COHORTS, GEN, generation_inputs, model_dir

MODEL = "Qwen/Qwen3-4B"
ROOT = Path(__file__).resolve().parents[2]
D = model_dir(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
rows = [json.loads(line) for line in COHORT.open()][COHORTS["dev"]]
bare = {r["scenario"]: r["text"] for r in map(json.loads, (D / "answers/bare/bare.jsonl").open())}
texts = [p + bare[r["scenario"]] for p, r in zip(generation_inputs(tok, rows), rows)]
vec = Vector.load(str(D / "vectors/sink_value_s0.safetensors"))

vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
punct = torch.tensor([any(c in t for c in ".\n,;:!?") and len(t.strip("Ġ Ċ.,;:!?\n")) == 0 for t in vocab], device="cuda")
cfg = model.config
H, KVH, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
layers = model.model.layers
stats = {L: {"a_first": torch.zeros(H), "a_punct": torch.zeros(H), "write": 0.0} for L in range(len(layers))}
n = 0
with torch.no_grad():
    for text in texts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        out = model(ids, output_attentions=True)
        is_p = punct[ids[0]].clone()
        is_p[0] = False
        for L, A in enumerate(out.attentions):  # A: [1, H, T, T]
            A = A[0, :, 16:].float()  # query positions ≥ 16
            stats[L]["a_first"] += A[:, :, 0].mean(1).cpu()
            stats[L]["a_punct"] += A[:, :, is_p].sum(-1).mean(1).cpu()
            if L in vec.stacked:
                v = vec.stacked[L]["x"].sum(0).view(KVH, d).cuda().float()
                W = layers[L].self_attn.o_proj.weight.float().view(-1, H, d)  # [D, H, d]
                per_head = torch.einsum("Dhd,hd->hD", W, v.repeat_interleave(H // KVH, 0))  # W_O^h v̂*_g(h)
                stats[L]["write"] += (A[:, :, 0].T @ per_head).norm(dim=-1).mean().item()  # [t, D] -> mean ‖·‖
                stats[L]["write_all"] = per_head.sum(0).norm().item()
        n += 1

table = []
for L, s in stats.items():
    af, ap = s["a_first"] / n, s["a_punct"] / n
    table.append({"layer": L, "sink attn, mean over heads": af.mean().item(), "sink attn, max head": af.max().item(),
                  "heads with sink ≥ 0.5": int((af >= 0.5).sum()), "punct attn, mean": ap.mean().item(),
                  "write at C=1": s["write"] / n, "write if all heads read sink": s.get("write_all", float("nan"))})
md = tabulate(table, headers="keys", tablefmt="pipe", floatfmt=".3f")
logger.info("SHOULD: sink attention high (>0.3) in most layers after the first few; write ≈ sink attn × write_all\n{}", md)
out_path = ROOT / "outputs/bsbench/results/q3-4b-dev/sink_probe.md"
out_path.write_text(f"# sink_probe: {MODEL}, 20 dev prompts + bare answers, query positions ≥ 16\n\n{md}\n")
