# Follow-up 3: engineered C1 cache mismatch — is the bounded dev run justified?

## Observed (log + diagnostic JSON)

- In one process (L40S, `Qwen3_5ForCausalLM`, `fla=True`, causal-conv1d falling back to torch): `fresh_vs_fresh_scaled=0`, `fresh_vs_repeat=0` on all 20; 3-prompt preflight `C1_logits=exact C1_greedy_ids=exact`, ordinary-vs-ordinary repeat exact.
- Across processes: `historical_vs_scaled=8`, `fresh_vs_scaled=11`, and historical≠fresh too (e.g. `leg_pnf_01`, `fin_pnf_01`). Every differing pair shares a long identical prefix then diverges mid-sentence.
- `C4_max_logit_delta` 18.125 vs 17.875 across processes = 2 bf16 ulps at that magnitude.

## Inference

The scaling code is not implicated: the only test that isolates it (paired, same process, full 512-token batch-20 generation) is 20/20 identical. What differs is *ordinary* generation across processes, at ulp scale, amplified by greedy near-ties. This is a property of the whole benchmark (every steering walk's answers were also generated in a different process from `bare.jsonl`), not of this method.

## Ranked likely causes

1. **Triton autotune in FLA gated-deltanet kernels**: config chosen by per-process timing, then cached → bit-stable within a process, different reduction order across processes. Fits the pattern exactly. Disprove: set a fixed autotune config / `TRITON_CACHE_DIR` shared, or run two processes back-to-back on the same container and compare 3-prompt logits.
2. **cuBLAS/SDPA algorithm selection** varying with workspace/memory state across processes. Disprove: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, compare.
3. **Environment drift between the historical run and now** (transformers/FLA versions, unpinned weights: `model_revision=None`). Missing evidence: image digest for the historical `prompting_engineered` run.
4. **Batch composition/padding** differs only if the historical run had partial-cache batches; probably not (20 missing each time), low rank.

## Verdict

Evidence justifies the bounded dev run. No unaddressed correctness blocker in the intervention; the failed assertion was a confounded check (cross-process cache identity), and replacing it with in-process paired identity is the right fix. Conditions:

- Run the paired check for **−C** too (the assert aborted before it).
- The sweep's C=1 point must be the in-process scaled answers; document that the plotted ★ (`prompting_engineered`) differs from sweep C=1 by drift only.
- Record per-rung run/process id and library versions in the certificate; resumed runs will mix processes.
- **Interpretation guard (cheap, judge-only, no GPU):** Jev-score the `fresh` vs `historical` pairs from the diagnostic JSON. That gives the effect/off-axis noise floor of zero-intervention drift on this cohort; any sweep point within it is not a finding. Without this, 8–11/20 textual changes under no intervention have unknown score impact.