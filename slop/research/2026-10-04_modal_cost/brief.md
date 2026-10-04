We run activation-steering dose sweeps on Modal and want them 3-10x cheaper. Answer in under one page, concrete and ranked by saving per hour of work.

Setup now: Qwen/Qwen3.5-4B (hybrid: Gated DeltaNet linear-attention layers + full attention), HF transformers 5.12 `model.generate`, bf16, greedy, left padding, batch 32, max_new_tokens 512. Steering = forward hooks that add a vector to the residual stream at chosen layers (sometimes only at the user-message token positions), so vLLM/SGLang need custom hooks. One "walk" = 14-21 doses x 2 signs x 200 prompts (~120-token prompts, answers usually 50-90 words, broken answers run to the token limit). ~115 s per dose on one L40S ($1.95/h). 34 walks cost 31 GPU-hours. Modal rates/h: T4 0.59, L4 0.80, A10 1.10, L40S 1.95, A100-40 2.10, H100 3.95. Modal starter plan: 10 GPUs concurrent.

transformers logs: "The fast path is not available because one of the required library is not installed. Falling back to torch implementation." flash-linear-attention 0.5.2 is installed; we suspect causal-conv1d is missing. Image: modal.Image.debian_slim(python 3.13).uv_pip_install(...), torch 2.11 cu130.

Questions:
1. Which GPU and batch size for a 4B bf16 model with these short generations? How to check we fill it?
2. How to get the Qwen3.5 fast path on Modal (causal-conv1d wheels for torch 2.11/cu130/py3.13, or a prebuilt image)?
3. Other big wins: stop broken answers early (repetition stop criteria), max_new_tokens, sorting prompts by length, several doses in one batch (different steering vector per batch row), vLLM with steering, fewer doses (adaptive search), sharing prefill KV across doses (steering only after the prompt?).
4. A tyro config layout: one subconfig per model size (model, batch, max_new_tokens, GPU). What fields matter?
State your confidence for each claim.
