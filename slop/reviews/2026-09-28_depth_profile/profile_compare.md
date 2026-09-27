Is the depth where the persona contrast forms a fixed % of depth? (PI/Claude, 2026-09-28)
Input: `walk.py --profile` log rows (outputs/logs/profile-{4b,27b,olmo}.log). ratio = |mean(h_pos) - mean(h_neg)| / mean |h| at the
last token, per layer. Output: profile_compare.md (crossing depths) and profile_compare.png (ratio / max vs depth).

| model | layers | peak layer (depth) | depth at 50% of peak | depth at 90% of peak | layers from end at 90% | steered band (depth) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 32 | L30 (0.97) | 0.52 (L16) | 0.61 (L19) | 12 | L6-L24 (0.19-0.77) |
| Qwen3.5-27B | 64 | L62 (0.98) | 0.78 (L49) | 0.98 (L62) | 1 | L12-L50 (0.19-0.79) |
| OLMo-2-32B | 64 | L47 (0.75) | 0.65 (L41) | 0.71 (L45) | 18 | L12-L50 (0.19-0.79) |
