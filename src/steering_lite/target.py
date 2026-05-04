"""Find transformer blocks (or sub-Linears) to hook.

Default: hook each block's forward output (residual stream after attn+mlp).
When `cfg.target_submodule` is set, it is interpreted as a **regex** matched
against `block.named_modules()` paths under each selected block; matching
`nn.Linear`s become the actual hook targets. This lets a single cfg target
multiple Linears per block (e.g. residual writers `mlp.down_proj` AND
`self_attn.o_proj`, or all 7 Linears in q/k/v/o/gate/up/down).

Works with HF llama-family architectures (llama, qwen, mistral, etc). For other
architectures, set `cfg.layers` to indices into whatever list lives at the path
your model exposes -- override `_get_blocks` if needed.
"""
import re
from torch import nn


def _get_blocks(model: nn.Module) -> nn.ModuleList:
    # llama-family: model.model.layers
    # gemma3-multimodal: model.language_model.layers (or model.model.language_model.layers)
    candidates = []
    inner = getattr(model, "model", model)
    candidates.append(inner)
    lm = getattr(inner, "language_model", None)
    if lm is not None:
        candidates.append(lm)
        candidates.append(getattr(lm, "model", lm))
    for c in candidates:
        blocks = getattr(c, "layers", None)
        if blocks is not None:
            return blocks
    raise RuntimeError(
        f"could not find .layers on {type(model).__name__}; "
        f"override _get_blocks for non-llama architectures"
    )


def find_targets(model: nn.Module, cfg) -> list[tuple[str, nn.Module, int]]:
    """Return [(full_name, module, layer_idx)] for hook targets selected by cfg.

    - `cfg.target_submodule is None`: one entry per selected block (the block itself).
    - `cfg.target_submodule = <regex>`: one entry per (block, matching nn.Linear).
      Regex is matched with `re.fullmatch` against `name` from `block.named_modules()`,
      e.g. `r"mlp\\.down_proj|self_attn\\.o_proj"` matches both residual writers,
      `r"self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj"` matches all 7 Linears.
    """
    blocks = _get_blocks(model)
    n = len(blocks)
    if cfg.layers is None:
        idxs = tuple(range(n))
    else:
        idxs = tuple(cfg.layers)
        for i in idxs:
            if not (0 <= i < n):
                raise ValueError(f"layer {i} out of range [0, {n})")
    sub = getattr(cfg, "target_submodule", None)
    if sub is None:
        return [(f"layers.{i}", blocks[i], i) for i in idxs]
    pattern = re.compile(sub)
    out = []
    for i in idxs:
        for name, mod in blocks[i].named_modules():
            if name and pattern.fullmatch(name) and isinstance(mod, nn.Linear):
                out.append((f"layers.{i}.{name}", mod, i))
    if not out:
        raise RuntimeError(
            f"target_submodule regex {sub!r} matched no nn.Linear "
            f"in {len(idxs)} selected blocks"
        )
    return out


def find_residual_linears(
    model: nn.Module,
    layer_indices: tuple[int, ...] | None = None,
    *,
    role: str = "both",                         # "writer" | "reader" | "both"
    fallback_regex: str | None = None,
) -> list[tuple[str, nn.Module, int, str]]:
    """Find Linears connected to the residual stream, per block.

    Returns `[(full_name, module, layer_idx, role)]` where role is "writer"
    (d_out == d_model, d_in != d_model) or "reader" (d_in == d_model,
    d_out != d_model). Square Linears are ambiguous (could be either) and
    are excluded by shape detection.

    Detection:
      1. Primary: weight.shape vs d_model.
      2. Fallback: if shape detection finds nothing (non-llama arch, weird
         shapes), match `fallback_regex` against `named_modules` paths and
         guess role from substring. Default regex covers llama-family names.
         Warns when fallback fires.
    """
    from loguru import logger
    d_model = get_d_model(model)
    blocks = _get_blocks(model)
    n = len(blocks)
    idxs = tuple(layer_indices) if layer_indices is not None else tuple(range(n))

    out: list[tuple[str, nn.Module, int, str]] = []
    for li in idxs:
        for name, mod in blocks[li].named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            d_out, d_in = mod.weight.shape
            is_writer = d_out == d_model and d_in != d_model
            is_reader = d_in == d_model and d_out != d_model
            if is_writer and role in ("writer", "both"):
                out.append((f"layers.{li}.{name}", mod, li, "writer"))
            elif is_reader and role in ("reader", "both"):
                out.append((f"layers.{li}.{name}", mod, li, "reader"))

    if out:
        return out

    regex = fallback_regex or r"mlp\.(down|gate|up)_proj|self_attn\.(q|k|v|o)_proj"
    logger.warning(
        f"shape-based residual-linear detection found nothing for d_model={d_model} "
        f"in {len(idxs)} blocks; falling back to regex {regex!r}"
    )
    pattern = re.compile(regex)
    writer_hints = ("down_proj", "o_proj")
    for li in idxs:
        for name, mod in blocks[li].named_modules():
            if not (name and pattern.fullmatch(name) and isinstance(mod, nn.Linear)):
                continue
            role_guess = "writer" if any(h in name for h in writer_hints) else "reader"
            if role in ("both", role_guess):
                out.append((f"layers.{li}.{name}", mod, li, role_guess))

    if not out:
        logger.warning(
            f"regex fallback {regex!r} also matched no Linears in layers {idxs}; "
            "super_sspace will have an empty basis"
        )
    return out


def get_d_model(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    d = getattr(cfg, "hidden_size", None)
    if d is None:
        # multimodal configs (gemma3): text sub-config
        text_cfg = getattr(cfg, "text_config", None)
        d = getattr(text_cfg, "hidden_size", None)
    if d is None:
        raise RuntimeError("model has no .config.hidden_size")
    return int(d)
