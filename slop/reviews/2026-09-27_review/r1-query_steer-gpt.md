**P1 — Default `QuerySteerC()` violates the all-layers contract.**  
`src/steering_lite/variants/query_steer.py:75` executes `"layers = tuple(cfg.layers)"`, but its inherited default is `None`; `src/steering_lite/config.py:20–21` explicitly promises `"# None = all layers"`. `train()` resolves targets but passes the unchanged config into prompt-based extraction.

Affected input: `sl.train(model, tok, POS, NEG, sl.QuerySteerC(), ...)` on supported, all-full-attention Qwen3. Static consequence: `TypeError: 'NoneType' object is not iterable`, before query extraction. Existing tests always supply explicit layers, masking this case. Resolve `None` to model block indices; hybrid models can retain their documented explicit-layer requirement.

Disproving check: run the existing query-steer pipeline with `QuerySteerC(layers=None)` on its tiny Qwen3 fixture and show successful extraction. Not executed here: no shell tool was available.

**Other inspected paths:** Qwen3 and Qwen3.5 expose post-normalization queries as `[batch, sequence, heads, head_dim]`; Qwen3.5 splits its gate before `q_norm`, so this hook does not modify gate channels. `_encode` explicitly requests right padding, matching `sum(mask)-1`. Runtime coefficient reads occur inside the hook; extraction hooks use `finally` removal, and installed hooks participate in ordinary detach. Registration and serialized query tensor shapes match the runtime.

**Evidence limits:** Supplied logs report `"51 passed in 38.60s"` and `"SMOKE_PASS method=query_steer rungs=2"`. These are pipeline evidence, not independent numerical verification of the query formula or scientific effectiveness. No additional concrete hook, save/load, or layer-filter defect established.

Merge verdict: OK with notes.