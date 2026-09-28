# VJP mechanism check (PI/Claude, 2026-09-28)

Steer with the stored seed-0 vector at C = +-C0/8 (C0 = 1-nat iso-KL coefficient), read the target-layer residual at the last token.
c = mean(h_pos) - mean(h_neg) of 32 held-out persona pairs at the target layer (the VJP cotangent).
cos = cos(mean shift, c); gain = mean shift . c_hat / |c| (share of the pos-neg gap moved along c). Base prompts: the 32 negative
persona prompts, or the first 32 benchmark prompts. Source logs: outputs/logs/vjpcheck-{4b,27b,olmo,27b-t48,olmo-t47}.log
(all doses 1/8..1 x C0 are in the logs). 4B random was SIGKILLed on Modal (exit -9) and not rerun.

| model | method | target | base | cos at +C0/8 | cos at -C0/8 | gain at +C0/8 | gain at -C0/8 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | vjp_delta | L29 | bench | -0.067 | +0.077 | -0.030 | +0.036 |
| Qwen3.5-4B | vjp_delta | L29 | persona_neg | -0.499 | +0.474 | -0.362 | +0.228 |
| Qwen3.5-4B | vjp_cache | L29 | bench | +0.011 | +0.016 | +0.007 | +0.010 |
| Qwen3.5-4B | vjp_cache | L29 | persona_neg | -0.445 | +0.385 | -0.380 | +0.207 |
| Qwen3.5-4B | mean_diff | L29 | bench | +0.443 | -0.449 | +0.260 | -0.266 |
| Qwen3.5-4B | mean_diff | L29 | persona_neg | +0.759 | -0.751 | +0.316 | -0.322 |
| Qwen3.5-27B | vjp_delta | L61 | bench | +0.001 | -0.003 | +0.000 | -0.001 |
| Qwen3.5-27B | vjp_delta | L61 | persona_neg | -0.347 | +0.358 | -0.300 | +0.225 |
| Qwen3.5-27B | vjp_cache | L61 | bench | -0.017 | +0.027 | -0.008 | +0.015 |
| Qwen3.5-27B | vjp_cache | L61 | persona_neg | -0.379 | +0.354 | -0.345 | +0.218 |
| Qwen3.5-27B | random | L61 | bench | +0.023 | +0.005 | +0.008 | +0.002 |
| Qwen3.5-27B | random | L61 | persona_neg | -0.012 | +0.046 | -0.002 | +0.009 |
| Qwen3.5-27B | mean_diff | L61 | bench | +0.291 | -0.278 | +0.177 | -0.180 |
| Qwen3.5-27B | mean_diff | L61 | persona_neg | +0.569 | -0.525 | +0.194 | -0.175 |
| OLMo-2-32B | vjp_cache | L61 | bench | +0.081 | -0.092 | +0.011 | -0.010 |
| OLMo-2-32B | vjp_cache | L61 | persona_neg | -0.441 | +0.487 | -0.143 | +0.152 |
| OLMo-2-32B | vjp_delta | L61 | bench | +0.090 | -0.077 | +0.003 | -0.002 |
| OLMo-2-32B | vjp_delta | L61 | persona_neg | -0.458 | +0.458 | -0.025 | +0.025 |
| OLMo-2-32B | mean_diff | L61 | bench | +0.461 | -0.455 | +0.065 | -0.066 |
| OLMo-2-32B | mean_diff | L61 | persona_neg | +0.610 | -0.582 | +0.065 | -0.060 |
| OLMo-2-32B | random | L61 | bench | +0.047 | -0.041 | +0.002 | -0.002 |
| OLMo-2-32B | random | L61 | persona_neg | +0.065 | -0.066 | +0.003 | -0.003 |
| Qwen3.5-27B | vjp_delta-t48 | L48 | bench | +0.001 | -0.011 | +0.001 | -0.008 |
| Qwen3.5-27B | vjp_delta-t48 | L48 | persona_neg | -0.170 | +0.313 | -0.222 | +0.323 |
| OLMo-2-32B | vjp_delta-t47 | L47 | bench | +0.012 | -0.004 | +0.000 | -0.000 |
| OLMo-2-32B | vjp_delta-t47 | L47 | persona_neg | -0.561 | +0.552 | -0.039 | +0.038 |
