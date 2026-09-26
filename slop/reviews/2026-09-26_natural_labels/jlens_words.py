"""Which words each method pushes, read from the residual change it causes (PI/Claude, 2026-09-26).

Method-independent: every method is attached at its Pareto-best C (full cohort, seed 0) and run on the same
fixed text (chat prompt + bare answer, teacher forced) for N_Q questions. Δh = mean over answer tokens and
questions of h_steered - h_bare. Readouts:
  J-lens @ layer MID: unembed(J_MID @ Δh)   (neuronpedia/jacobian-lens, Qwen3.5-4B, wikitext n1000)
  final layer:        unembed(Δh_final)      (plain logit lens; J is the identity there)
Top tokens up and down. Output: jlens_words.md next to this file.

Run (GPU): cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-26_natural_labels/jlens_words.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "/workspace/2026/jspace/j-steer-dev/docs/vendor/jacobian-lens")
import torch
import jlens
from loguru import logger
from steering_lite import Vector
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import COHORTS, ROOT, default_model_dir, load_cohort, read_answers
from walk import generation_inputs

MODEL = "Qwen/Qwen3.5-4B"
LENS = Path.home() / ".cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
N_Q, MID, TOP = 32, 24, 12

md = default_model_dir()
site = json.loads((ROOT / "outputs/bsbench/results/full/points.json").read_text())
cohort = load_cohort()
scenarios = list(cohort)[COHORTS["full"]][:N_Q]
bare = read_answers(md / "answers/bare/bare.jsonl")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
unembed = jlens.from_hf(model, tok).unembed
lens = jlens.JacobianLens.load(str(LENS))
layers = model.model.layers
final = len(layers) - 1

prompts = generation_inputs(tok, [cohort[s] for s in scenarios])
texts = [p + bare[s]["text"] for p, s in zip(prompts, scenarios)]
n_prompt = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]


@torch.inference_mode()
def mean_resid() -> dict[int, torch.Tensor]:
    """Mean residual over answer tokens and questions at MID and final (one question at a time: no padding)."""
    sums = {MID: 0, final: 0}
    for text, n in zip(texts, n_prompt):
        grab = {}
        hooks = [layers[l].register_forward_hook(lambda m, i, o, l=l: grab.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, n:].float().mean(0))) for l in sums]
        model(**tok(text, return_tensors="pt", add_special_tokens=False).to("cuda"))
        for h in hooks:
            h.remove()
        for l in sums:
            sums[l] = sums[l] + grab[l] / len(texts)
    return sums


def words(delta: torch.Tensor, layer: int) -> tuple[list[str], list[str]]:
    logits = unembed(lens.transport(delta.unsqueeze(0), layer) if layer != final else delta.unsqueeze(0))[0].float()
    logits = logits - logits.mean()
    fmt = lambda idx: [repr(tok.decode([int(i)]).strip())[1:-1] for i in idx]
    return fmt(logits.topk(TOP).indices), fmt((-logits).topk(TOP).indices)


h_bare = mean_resid()
lines = [__doc__, f"questions: {N_Q} (first of the full cohort), MID layer {MID}, final layer {final}\n",
         "| method | side | C | J-lens @MID: up | J-lens @MID: down | final: up |", "|---|---|---|---|---|---|"]
for row in site["summary"]:
    method = row["method"]
    if method.startswith("prompting"):
        continue
    vector = Vector.load(str(md / "vectors" / f"{method}_s0.safetensors"))
    for side in ("+C", "-C"):
        best = row["best"][side]
        if best is None:
            continue
        sign = 1 if side == "+C" else -1
        with vector(model, C=sign * best["C"]):
            h = mean_resid()
        up, down = words(h[MID] - h_bare[MID], MID)
        fup, _ = words(h[final] - h_bare[final], final)
        lines.append(f"| {method} | {side} | {best['C']:.3g} | {' '.join(up)} | {' '.join(down)} | {' '.join(fup)} |")
        logger.info(lines[-1])
    del vector
text = "\n".join(lines)
(Path(__file__).parent / "jlens_words.md").write_text(text + "\n")
print(text)
