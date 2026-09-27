"""Vector geometry per model (PI/Claude, 2026-09-27): cross-seed cos per layer, cos(vjp_delta, mean_diff), and the share of
squared norm in the top-1 / top-10 coordinates (a vector dominated by a few outlier channels would score high).
Run: .venv/bin/python slop/reviews/2026-09-27_27b_debug/vector_geometry.py > slop/reviews/2026-09-27_27b_debug/vector_geometry.md
"""
import glob, itertools
from safetensors.torch import load_file

def vecs(md, meth, s):
    d = load_file(f"{md}/vectors/{meth}_s{s}.safetensors")
    return {int(k.split(".")[1].replace("layer", "")): v.float().flatten() for k, v in d.items() if k.startswith("stacked.") and k.split(".")[-1] in ("v", "c")}
cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
print("| model | method | seeds | cross-seed cos mean (min) | top-1 share | top-10 share | cos(., mean_diff) same layer |\n|---|---|---|---|---|---|---|")
for name, pat in (("Qwen3.5-4B", "Qwen--Qwen3.5-4B"), ("Qwen3.5-27B", "Qwen--Qwen3.5-27B"), ("OLMo-2-32B", "allenai--OLMo-2-0325-32B-Instruct")):
    md = glob.glob(f"outputs/bsbench/{pat}-g*")[0]
    for m in ("mean_diff", "vjp_delta", "vjp_cache", "random"):
        seeds = [s for s in (0, 1, 2) if glob.glob(f"{md}/vectors/{m}_s{s}.safetensors")]
        V = [vecs(md, m, s) for s in seeds]
        L = sorted(V[0])
        cs = [cos(V[a][l], V[b][l]) for a, b in itertools.combinations(range(len(V)), 2) for l in L]
        e = [(V[0][l] ** 2 / (V[0][l] ** 2).sum()).sort(descending=True).values for l in L]
        t1, t10 = sum(float(x[0]) for x in e) / len(e), sum(float(x[:10].sum()) for x in e) / len(e)
        md0 = vecs(md, "mean_diff", 0)
        cm = [cos(V[0][l], md0[l]) for l in L if l in md0 and V[0][l].shape == md0[l].shape]
        xs = f"{sum(cs)/len(cs):+.3f} ({min(cs):+.3f})" if cs else "1 seed"
        print(f"| {name} | {m} | {len(seeds)} | {xs} | {t1:.3f} | {t10:.3f} | {sum(cm)/len(cm):+.3f} |" if cm else f"| {name} | {m} | {len(seeds)} | {xs} | {t1:.3f} | {t10:.3f} | n/a |")
