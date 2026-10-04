import json
from pathlib import Path
from statistics import mean
from data import load_cohort, read_answers
from judge import bsb_request, cached, key
D=Path("../../slop/reviews/2026-10-04_eval_v3/")
M=Path("../../outputs/bsbench/Qwen--Qwen3.5-4B-g1f092bc2/answers/")
C={"bare":M/"bare/bare.jsonl","prompt -C":M/"prompting_s0/-C_C1.jsonl","mean_diff -C 0.5":M/"mean_diff_s0/-C_C0.5.jsonl","vjp_resid -C 0.198":M/"vjp_resid_s0/-C_C0.1984251315.jsonl"}
import sys
C |= {arg.split("=",1)[0]: Path("../..")/arg.split("=",1)[1] for arg in sys.argv[1:]}
rows={(r["condition"],r["scenario"]):r for r in map(json.loads,open(D/"regrade_sonnet.jsonl"))}
qs=load_cohort(); have=cached(); ans={c:read_answers(p) for c,p in C.items()}
sample=sorted({s for _,s in rows if all((c,s) in rows for c in C)})
jev=lambda c,s: have[key(bsb_request(qs[s]["prompt"],qs[s]["nonsensical_element"],ans[c][s]["text"]))]["bs_score"]["score"]
print(f"# Sonnet 4.6 (BullshitBench panel judge, their prompt, 0/1/2) vs Jev on the -C side — PI/OpenAI 2026-10-04\n\nn = {len(sample)} questions (random 40 of 100, seed 20261004; one dropped: OpenRouter credits ran out). Answers from outputs/bsbench/Qwen--Qwen3.5-4B-g1f092bc2.\n")
print("| condition | Sonnet mean | Jev mean | Sonnet gain vs bare | Jev gain vs bare | Sonnet: share 2 | share 0 | mean abs(Jev − Sonnet) |")
print("|---|---|---|---|---|---|---|---|")
bs=mean(rows["bare",s]["score"] for s in sample); bj=mean(jev("bare",s) for s in sample)
for c in C:
    S=[rows[c,s]["score"] for s in sample]; J=[jev(c,s) for s in sample]
    print(f"| {c} | {mean(S):.2f} | {mean(J):.2f} | {mean(S)-bs:+.2f} | {mean(J)-bj:+.2f} | {sum(x==2 for x in S)/len(S):.0%} | {sum(x==0 for x in S)/len(S):.0%} | {mean(abs(a-b) for a,b in zip(S,J)):.2f} |")
allS=[rows[c,s]["score"] for c in C for s in sample]; allJ=[jev(c,s) for c in C for s in sample]
ms,mj=mean(allS),mean(allJ); cov=sum((a-ms)*(b-mj) for a,b in zip(allS,allJ)); r=cov/(sum((a-ms)**2 for a in allS)*sum((b-mj)**2 for b in allJ))**.5
print(f"\nCorrelation Jev vs Sonnet over all {len(allS)} answers: r = {r:.2f}\n\nLargest disagreements (Jev − Sonnet):")
dis=sorted(((jev(c,s)-rows[c,s]["score"],c,s) for c in C for s in sample),key=lambda x:-abs(x[0]))
for d,c,s in dis[:8]:
    print(f"\n- [{c} | {s.replace('syco_bullshit_v2_','')}] Jev {jev(c,s):.2f} vs Sonnet {rows[c,s]['score']}. Sonnet: \"{rows[c,s]['justification']}\"\n  answer: \"{ans[c][s]['text'][:260]}\"")
