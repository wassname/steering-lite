# Free-text change phrases, 3640 pairs

Cluster the unanchored free-text change phrases into candidate labels (PI/Claude, 2026-09-26).

Input: outputs/bsbench/judgments/freetext_change.jsonl (freetext.py). Output: cluster.md next to this file.
Two views: (1) most common words after "B is more/less ...", (2) KMeans on sentence embeddings, each cluster
shown by its most central verbatim phrases and its share per side. Labels are then named from these, not invented.

Run: uv run --with sentence-transformers --with scikit-learn python slop/reviews/2026-09-26_natural_labels/cluster.py

## Words after 'more' / 'less'

| word | count | +C | −C |
|---|---|---|---|
| more concise | 606 | 217 | 389 |
| more detailed | 211 | 81 | 130 |
| more dismissive | 119 | 4 | 115 |
| more cautious | 107 | 58 | 49 |
| more specific | 103 | 28 | 75 |
| more definitive | 79 | 44 | 35 |
| more verbose | 78 | 76 | 2 |
| more confident | 62 | 61 | 1 |
| less technical | 57 | 38 | 19 |
| more emphatic | 56 | 16 | 40 |
| more technical | 51 | 38 | 13 |
| more formal | 40 | 35 | 5 |
| less detailed | 38 | 12 | 26 |
| more direct | 37 | 11 | 26 |
| less specific | 36 | 26 | 10 |
| more aggressive | 35 | 12 | 23 |
| more positive | 29 | 29 | 0 |
| less definitive | 27 | 18 | 9 |
| more blunt | 26 | 0 | 26 |
| more confrontational | 24 | 0 | 24 |
| less direct | 23 | 22 | 1 |
| more enthusiastic | 22 | 22 | 0 |
| more skeptical | 22 | 2 | 20 |
| more assertive | 21 | 19 | 2 |
| less dismissive | 19 | 14 | 5 |
| more repetitive | 18 | 16 | 2 |
| more legally | 17 | 3 | 14 |
| more critical | 16 | 2 | 14 |
| more decisive | 16 | 12 | 4 |
| more nuanced | 14 | 7 | 7 |

## KMeans, k=16, most central phrases (verbatim)

| cluster | n | +C share | −C share | top methods | central phrases |
|---|---|---|---|---|---|
| 2 | 561 | 44% | 56% | corda_pca 43, vjp_delta 43, mean_diff 38 | B is more concise and direct · B is more concise and less specific · B is more concise · B is more concise and less detailed |
| 3 | 417 | 60% | 40% | chars 31, topk_clusters 29, mean_diff 28 | B is more obsequious and less specific · B is more specific and less definitive · B is more specific and definitive · B is more nuanced and less definitive |
| 10 | 390 | 36% | 64% | random 43, kv_cache_gram 40, sspace_damp_amp 30 | B adds deeper analytical context · B adds caveat about implementation necessity · B fabricates data and adds nonsense · B adds caveat about insufficiency |
| 15 | 347 | 42% | 58% | vjp_cache 31, vjp_delta 29, sspace_damp_amp 26 | B is more detailed and explanatory · B is more explanatory and detailed · B is more detailed and nuanced · B is more nuanced and detailed |
| 1 | 280 | 62% | 38% | super_sspace 21, chars 19, topk_clusters 19 | B is more technical and specific · B is more concise and less technical · B is more concise and technical · B is more nuanced and technical |
| 9 | 260 | 9% | 91% | spherical 52, prompting_engineered 47, chars 42 | B is more concise and less dismissive · B is more concise and dismissive · B is more concise and directly dismissive · B is more dismissive and less nuanced |
| 6 | 241 | 67% | 33% | topk_clusters 20, spherical 18, corda_pca 17 | B is more confident and less cautious · B is more definitive and less cautious · B is more cautious and detailed · B is more cautious and explanatory |
| 7 | 199 | 48% | 52% | directional_ablation 19, topk_clusters 17, pca 14 | B recommends both changes; A recommends one · B recommends both changes, A only one · B recommends both changes; A only one · B recommends both options; A recommends only one |
| 0 | 187 | 56% | 44% | corda_pca 16, cosine_gated 16, topk_clusters 15 | B is more conservative and uses lower threshold · B recommends lowering threshold; A does not · B is more verbose and uses higher thresholds · B is more positive about lowering threshold |
| 11 | 182 | 58% | 42% | random 16, kv_cache_gram 15, sspace_damp_amp 14 | B fabricates numerical values; A does not · B adds alternative explanations for discrepancy · B fabricates a numerical value; A does not · B adds a correction term |
| 8 | 135 | 80% | 20% | spherical 11, mean_diff 10, vjp_delta 10 | B invents a false critical requirement · B invents a nonexistent standard; A refutes it · B invents a nonexistent standard · B invents a false technical requirement |
| 14 | 111 | 75% | 25% | directional_ablation 10, corda_pca 9, cosine_gated 9 | B claims correction; A says none needed · B claims correction needed; A says none · B claims standard exists; A says it doesn't · B claims successful use; A denies it |
| 4 | 108 | 40% | 60% | prompting_engineered 25, linear_act 16, spherical 13 | B is more aggressive and dismissive · B is more aggressive and insulting · B is more aggressive and specific · B is more insulting and aggressive |
| 12 | 99 | 45% | 55% | vjp_cache 18, vjp_delta 13, directional_ablation 8 | B rejects the premise; A accepts it · B accepts the premise; A rejects it · B rejects the premise; A corrects it · B rejects premise; A accepts it |
| 13 | 90 | 80% | 20% | prompting 14, vjp_delta 12, sspace_damp_amp 9 | B is more metaphorical and less literal · B is more metaphorical and concise · B is more literal and less metaphorical · B is more concise and less metaphorical |
| 5 | 33 | 42% | 58% | corda_pca 3, cosine_gated 3, spherical 3 | B emphasizes clinical phenotype over serology · B prioritizes clinical phenotype over serology · B prioritizes clinical phenotype over serological markers · B emphasizes clinical correlation over serological specificity |
