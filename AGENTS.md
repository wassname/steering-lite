# AGENTS.md

Inherits conventions from sibling project `lora-lite`. Read [../lora-lite/AGENTS.md](../lora-lite/AGENTS.md) if it exists.

## House rules

- Fail fast. No defensive programming, no fallbacks, no silent dequant.
- One file per method under `src/steering_lite/variants/`. Docstring -> paper link, math, intuition.
- Use `einops.einsum` and `jaxtyping` shape annotations. Tensor dim letters: `b s d` (batch, seq, d_model), `n` (prompts), `r` (rank/components), `k` (clusters), `l` (layer).
- No backward compat. Break things to gain simplicity.
- Single functional test = the real benchmark at tiny scale. Don't add a separate "unit test" suite.
- New methods register via `@register_config` and `@register` decorators; export `Config` class from `__init__.py`.

## Verify

`just smoke` -> 6/6 methods pass full extract -> attach -> generate -> save/load loop on tiny-random-Llama. Asserts non-zero state, nonzero residual delta, save/load round-trip < 1e-4.

`just bench MODEL METHOD COEFF` -> JSON to `outputs/daily_dilemmas/` with surgical informedness.
