# Steering-lite BS-bench rewrite

> User: "Just use this instead of what we have. a rewrite. on a branch."
> User: "It's a first pass suitable for use and demo."
> User: "spend <$50 on this initial set."

- [/] goal: reuse vjp-steering as the benchmark base on `rewrite/bsbench-vjp`
  - Inspect source, judge payloads, costs and cache contracts before changing them.
  - Preserve steering-lite methods; replace the moral-foundation benchmark entrypoint with numbered BS-bench v2, initially 20 questions and Qwen3.5-4B.
  - Failure: a copied framework brings unrelated experiments or silently changes the reference protocol.
  - Deliverable: one `just sweep` entrypoint, pinned source provenance, minimal imported components.
- [ ] goal: compare prompting, random directions, mean difference, PCA, and VJP methods
  - Reuse existing VJP-delta semantics; resolve VJP-diff naming from source; add cache-target VJPs with explicit gradient path.
  - Validate prefix/suffix examples and paired persona behavior using minimal existing library functionality and the existing judge.
  - Failure: a cache gradient is detached or pairs steer a different concept; tests must distinguish nonzero gradients from faithful extraction.
  - Deliverable: tiny-model extract/attach/generate/save-load checks plus paired examples and judge disagreements.
- [ ] goal: find useful doses cheaply and check KL-target transfer on about four cases
  - Reuse iterative dose search; measure a method/model-specific RMS token-KL target at a usable dose and test it on new inputs.
  - Keep this a cheap development experiment, not a publication-validation study.
  - Failure: zeros trivially look coherent, or the same inputs are reused as transfer cases.
  - Deliverable: predicted/observed doses, coherence and effect with complete examples; diagnose failures before changing the method.
- [ ] goal: retain judged effects and add target-blind change descriptions
  - Keep the existing paired base/steered on/off-target judge; separately ask for unlabeled changes with descriptions and magnitudes.
  - Failure: target, method or sign leaks into the blind request, or stale judge responses are reused.
  - Deliverable: numbered paired outputs, full request/response evidence, prompt-content-keyed cache.
- [ ] goal: cached Modal sweep below $50 including GPU and judging
  - Reuse measured costs; run real-pipeline smoke first, then a bounded initial inventory and track spending.
  - Failure: reruns silently regenerate or cache identity omits a consequential parameter.
  - Deliverable: rerun demonstrates cache hits; changed inputs invalidate relevant stages; cost ledger and run logs.
- [ ] goal: reference-style HTML/PNG plots and max-dose/optimal-dose tables
  - Keep the random region and Pareto curves; use working utility `directed effect - 4 * off-target effect` for dose selection, separately from maximum coherent dose.
  - Failure: static and interactive views differ or old measurements appear as new results.
  - Deliverable: inspect both reference/new PNGs and obtain fresh-eyes review; identical source points feed HTML, PNG and tables.

## UAT / Verification
- Success: cached `just sweep` produces numbered real generations, judged effects/blind descriptions, plot and dose tables within budget.
- Likely failure: model/cache incompatibility or incoherent steering; inspect full log and paired outputs, isolate/fix and rerun at small scale.
- Subtle failure: plausible scores from stale data or target leakage; validate content identities, request payloads and plot/table point parity.
- Scaling beyond 4B and the 100-question evaluation wait until the initial development comparison is interpretable.

-- PI/OpenAI
