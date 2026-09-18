# KV-cache Gram leaderboard prediction

Question: does a contrastive direction in the high-energy value-cache subspace steer the moral axis at matched output-distribution drift without losing more selectivity than residual methods?

Novel part: the intervention changes actual cached values per KV head. Extraction uses no gradients or optimizer. The existing bare, prompt-only, residual PCA, CorDA, S-space, and random rows are the controls.

| outcome | prior | distinguishing result |
|---|---:|---|
| It steers the intended axis with usable selectivity | 40% | positive held-out steering selectivity and intended movement at calibrated KL |
| It changes behaviour but loses selectivity | 30% | intended movement with comparable or larger unintended movement |
| It is too weak after rank-16 projection | 15% | calibration reaches its coefficient ceiling below the KL target |
| It becomes incoherent before useful movement | 10% | calibration trace shows repetition or broken generations before axis movement |
| Implementation or evaluation bug | 5% | impossible cache shapes, non-finite values, cache path mismatch, or failure of saved software invariants |

I would drop this exact rank-16 implementation if it has no intended movement at the largest coherent calibrated coefficient while the existing methods move on the same evaluation. That would not rule out other cache edit rules, key editing, cache VJP extraction, or different ranks.

Expected runtime cost: fitting performs ordinary cached forwards plus CPU float64 Gram accumulation and per-head eigendecomposition. Inference adds one rank-independent projection per stored value and extracted direction at selected full-attention layers. No backward graph is retained.

The result must be audited from the complete calibration and generation logs before adding a numeric README row.

Queued as pueue job 1704 from commit `efd74bb`. It runs only `kv_cache_gram` on `Qwen/Qwen3-4B` with rank 16 and the existing full-sweep defaults (`target_kl=0.8`, `kl_rms`, 256 contrast pairs, classic vignettes). The prompt-only baseline is omitted because the comparable frozen sweep already has it; this job still recomputes bare.

Result: credible negative for this implementation. The canonical selectivity was -1.16 (95% interval [-1.85,-0.45]); +C selected the `social` / not-morally-wrong category on 115/132 rows. The README row condition was not met. See [job 1704 audit](../../docs/audits/job_1704.md).

-- PI/OpenAI
