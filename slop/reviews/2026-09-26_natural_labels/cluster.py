"""Cluster the unanchored free-text change phrases into candidate labels (PI/Claude, 2026-09-26).

Input: outputs/bsbench/judgments/freetext_change.jsonl (freetext.py). Output: cluster.md next to this file.
Two views: (1) most common words after "B is more/less ...", (2) KMeans on sentence embeddings, each cluster
shown by its most central verbatim phrases and its share per side. Labels are then named from these, not invented.

Run: uv run --with sentence-transformers --with scikit-learn python slop/reviews/2026-09-26_natural_labels/cluster.py
"""
import collections, json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

K = 16
here = Path(__file__).parent
rs = [json.loads(line) for line in open(here.parents[2] / "outputs/bsbench/judgments/freetext_change.jsonl")]
text = [r["change"].strip().rstrip(".") for r in rs]
lines = [f"# Free-text change phrases, {len(rs)} pairs\n", __doc__, "## Words after 'more' / 'less'\n", "| word | count | +C | −C |", "|---|---|---|---|"]
words = collections.Counter()
by_side = collections.defaultdict(collections.Counter)
for r, t in zip(rs, text):
    for w in re.findall(r"\b(more|less) (\w+)", t.lower()):
        words[" ".join(w)] += 1
        by_side[r["side"]][" ".join(w)] += 1
lines += [f"| {w} | {n} | {by_side['+C'][w]} | {by_side['-C'][w]} |" for w, n in words.most_common(30)]

emb = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(text, normalize_embeddings=True, batch_size=256)
km = KMeans(K, n_init=10, random_state=0).fit(emb)
lines += ["", f"## KMeans, k={K}, most central phrases (verbatim)\n", "| cluster | n | +C share | −C share | top methods | central phrases |", "|---|---|---|---|---|---|"]
for c in sorted(range(K), key=lambda c: -(km.labels_ == c).sum()):
    idx = np.where(km.labels_ == c)[0]
    d = np.linalg.norm(emb[idx] - km.cluster_centers_[c], axis=1)
    central = [text[i] for i in idx[np.argsort(d)]]
    seen, show = set(), []
    for t in central:
        if t.lower() not in seen:
            seen.add(t.lower()); show.append(t)
        if len(show) == 4:
            break
    sides = collections.Counter(rs[i]["side"] for i in idx)
    meth = collections.Counter(rs[i]["method"] for i in idx).most_common(3)
    lines.append(f"| {c} | {len(idx)} | {sides['+C'] / len(idx):.0%} | {sides['-C'] / len(idx):.0%} | {', '.join(f'{m} {n}' for m, n in meth)} | {' · '.join(show)} |")
out = "\n".join(lines)
print(out)
(here / "cluster.md").write_text(out + "\n")
