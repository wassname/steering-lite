# Steering method naming opinion

Prepared by PI/OpenAI. Read-only naming consultation, not an implementation or experiment audit.

User request: "we want names to distinguish them, be semantic, obvius, and fit with other steering mehtod if possibe". User also says "vjp-delta_value is a bit long hmm". They want names for their research methods, including VJP, CorDA-derived steering, S-space variants, and attention-sink query steering. Recommend a coherent short naming scheme, not a new acronym per method. Preserve attribution when adapting published methods. No code changes requested yet.

Evidence: source-docstrings.md in this directory copies all variant module docstrings verbatim. Read that file first; these are implementation-author descriptions, not independent verification. If a naming distinction depends on details absent from the docstrings, read the exact corresponding file under /workspace/2026/lite/steering-lite-bsbench/src/steering_lite/variants/. Do not search /workspace. All paths here are accessible with read. No experiments or ML run diagnosis needed.

Key facts checked by the parent:
- vjp_delta: c = mean positive target hidden states minus mean negative; per-layer direction = mean_positive(J^T c) minus mean_negative(J^T c), applied to residual activations. The reference repo calls it vjp_delta.
- vjp_cache uses the same estimator through actual cached VALUES, not keys; interventions edit values.
- corda_pca uses the CorDA decomposition algebra with regularized last-token context second moments, then PCA of paired differences in adapter hidden coordinates. It is an adaptation, not the original training algorithm.
- sspace uses an individual weight matrix's SVD coordinates, contrast direction, and cosine-dependent steering. super_sspace pools residual-side singular factors of writer/reader matrices to construct a shared residual basis.
- svdkv splits the runtime attention sink into two synthetic KV slots carrying opposite contrastive value directions and shifts queries to alter their relative attention. svdkv_resid combines this with mean_diff residual steering. Both directions/bases use decompositions but that alone may not uniquely identify the method.

Output approximately 500 words: give a concise naming principle; current -> recommended name with a short reason for the main families; identify any important misleading names among the remaining methods. Separate naming grounded in this supplied source from any claimed established field terminology (do not invent precedent). Favor minimal worthwhile changes and names readable in plot legends and Python module paths. Explicitly discuss whether VJP names should distinguish estimator or intervention location, and where brevity becomes ambiguous. Do not repeat performance claims. The parent's preferred replacements have intentionally been withheld to obtain an independent opinion.
