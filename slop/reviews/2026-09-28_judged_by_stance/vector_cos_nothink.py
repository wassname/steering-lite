"""Does dropping the literal "<think>" from the extraction pairs change the OLMo vjp_delta vector? (PI/Claude, 2026-09-28)

cos(v_default, v_nothink) per source layer, seed 0, from the saved vectors:
outputs/bsbench/allenai--OLMo-2-0325-32B-Instruct-g7c7712c6/vectors/vjp_delta{,-nothink}_s0.safetensors.
Writes vector_cos_nothink.md next to this file.
Run: .venv/bin/python slop/reviews/2026-09-28_judged_by_stance/vector_cos_nothink.py
"""
from pathlib import Path
from statistics import median

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "outputs/bsbench/allenai--OLMo-2-0325-32B-Instruct-g7c7712c6/vectors"
default = load_file(str(VECTORS / "vjp_delta_s0.safetensors"))
nothink = load_file(str(VECTORS / "vjp_delta-nothink_s0.safetensors"))
assert default.keys() == nothink.keys(), (sorted(default), sorted(nothink))
layer = lambda key: int(key.split(".")[1].removeprefix("layer"))  # stacked.layer<l>.v
cos = {layer(key): float(torch.nn.functional.cosine_similarity(default[key].float().flatten(), nothink[key].float().flatten(), dim=0))
       for key in default}
lines = ["| layer | cos(default, nothink) |", "|---|---|", *(f"| L{l} | {c:+.4f} |" for l, c in sorted(cos.items()))]
summary = f"{len(cos)} source layers: min {min(cos.values()):+.4f}, median {median(cos.values()):+.4f}, max {max(cos.values()):+.4f}"
text = __doc__ + "\n" + summary + "\n\n" + "\n".join(lines) + "\n"
(Path(__file__).parent / "vector_cos_nothink.md").write_text(text)
print(summary)
