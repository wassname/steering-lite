"""Scratch: is the chars/vjp_cache -C disagreement a Jev floor effect (bare already rejects the premise)?"""
import sys; sys.path.insert(0, ".")
from statistics import mean
from judge import default_model_dir
from results import build_points, jev_points, method_curve, pareto_score

EX = set("corda_pca directional_ablation linear_act sspace sspace_damp_amp sspace_pca super_sspace topk_clusters".split())
md = default_model_dir()
ds = build_points(md, "full", EX)
jv = jev_points(ds, md)
bare = {}
for p in jv:
    for q in p["questions"]:
        # bare Jev level = steered level - effect
        bare.setdefault(q["scenario"], round(float(q["evidence"].split("level ")[1].split(" ->")[0]), 2))
levels = list(bare.values())
print(f"bare Jev premise level over {len(levels)} questions: <=1 (rejects/flags) {sum(l <= 1 for l in levels)}, 1-3 {sum(1 < l <= 3 for l in levels)}, >3 (accepts) {sum(l > 3 for l in levels)}")
for m in ("chars", "vjp_cache", "spherical", "vjp_delta"):
    cd, cj = method_curve(ds, m, "-C"), method_curve(jv, m, "-C")
    bd = pareto_score({"-C": cd})[1]["-C"]
    idx = [i for i, p in enumerate(cd) if p["C"] == bd["C"]][0]
    qd, qj = cd[idx]["questions"], cj[idx]["questions"]
    for name, sel in (("bare at floor (Jev<=1)", lambda s: bare[s] <= 1), ("bare not at floor (Jev>1)", lambda s: bare[s] > 1)):
        d = [-a["effect"] for a in qd if sel(a["scenario"])]; j = [-a["effect"] for a in qj if sel(a["scenario"])]
        print(f"{m:10s} -C C={bd['C']:.3g} {name:26s} n={len(d):3d}  DeepSeek candour gain {mean(d):+.2f}  Jev premise drop {mean(j):+.2f}")

# read: floor questions where DeepSeek's chars-vs-vjp_cache -C gap is largest (seed 0)
def at(m, curves):
    c = method_curve(curves, m, "-C"); b = pareto_score({"-C": c})[1]["-C"]
    return {(q["scenario"], q["seed"]): q for q in [p for p in c if p["C"] == b["C"]][0]["questions"]}
ch, vc = at("chars", ds), at("vjp_cache", ds)
from judge import read_answers
bare_text = read_answers(md / "answers/bare/bare.jsonl")
gaps = sorted(((-ch[k]["effect"]) - (-vc[k]["effect"]), k) for k in ch if k[1] == 0 and bare[k[0]] <= 1)
for gap, k in gaps[-3:][::-1]:
    print(f"\n==== {k[0]}  DeepSeek chars-vjp_cache candour gap {gap:+.2f}")
    print("BARE:", bare_text[k[0]]["text"][:400].replace("\n", " "))
    for name, q in (("chars", ch[k]), ("vjp_cache", vc[k])):
        print(f"{name.upper()} (DeepSeek candour {-q['effect']:+.2f}):", q["text"][:400].replace("\n", " "), "\n   evidence:", q["evidence"])
