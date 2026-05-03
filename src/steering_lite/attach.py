"""attach / detach / save / load. The whole runtime.

Two hook paths:

  - Default: hook each transformer **block**'s forward output. The variant's
    `apply(block, h, state, cfg)` modifies the block's hidden_states.
  - Linear-IO (variants with `requires_linear_io = True`): hook a sub-module
    (e.g. `mlp.down_proj`) of each block. The variant's
    `apply(block, x, y, state, cfg)` sees both the sub-module's input x and
    output y, returns the modified output. State buffers still live on the
    parent block; the sub-module gets a backref via `_steering_block_ref`.
"""
from __future__ import annotations
import json
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .config import SteeringConfig, REGISTRY
from .target import find_targets
from .extract import record_activations
from .extract_attn import (
    record_activations_attn,
    record_activations_mean,
    Mode, PairAgg, Pool,
)


_ATTACHED_ATTR = "_steering_lite_attached"
_STATE_PREFIX = "_steering_state_"


def _gather_state(block) -> dict[str, torch.Tensor]:
    return {
        k[len(_STATE_PREFIX):]: getattr(block, k)
        for k in dir(block)
        if k.startswith(_STATE_PREFIX) and isinstance(getattr(block, k, None), torch.Tensor)
    }


def _hook(block, args, out):
    cfg: SteeringConfig = block._steering_cfg
    method = block._steering_method
    state = _gather_state(block)
    if isinstance(out, tuple):
        h = out[0]
        h_new = method.apply(block, h, state, cfg)
        return (h_new,) + out[1:]
    return method.apply(block, out, state, cfg)


def _linear_hook(mod, args, out):
    """Forward hook for sub-module variants (requires_linear_io=True).

    `out` is a Tensor (Linear's output), not a tuple. State is on the parent
    block, accessed via `mod._steering_block_ref`.
    """
    block = mod._steering_block_ref
    cfg: SteeringConfig = block._steering_cfg
    method = block._steering_method
    state = _gather_state(block)
    x = args[0]
    return method.apply(block, x, out, state, cfg)


def _install_state(block: nn.Module, state: dict[str, torch.Tensor], cfg: SteeringConfig) -> None:
    for k, v in state.items():
        attr = _STATE_PREFIX + k
        if hasattr(block, attr):
            raise RuntimeError(f"block already has {attr}; detach first")
        block.register_buffer(attr, v.to(cfg.dtype), persistent=True)


def attach(
    model: nn.Module,
    cfg: SteeringConfig,
    vectors,
) -> list[RemovableHandle]:
    """Install per-layer state as buffers and register forward hooks.

    `vectors` may be a `dict[int, dict[str, Tensor]]` or a `Vector` (unwrapped).
    Hook target depends on the method's `requires_linear_io` flag (sub-module
    forward_hook with input+output) vs the default block forward_hook.
    """
    from .vector import Vector
    if isinstance(vectors, Vector):
        vectors = vectors.state
    if cfg.method not in REGISTRY:
        raise KeyError(f"unknown method {cfg.method!r}; registered: {list(REGISTRY)}")
    method = REGISTRY[cfg.method]
    requires_linear = getattr(method, "requires_linear_io", False)
    targets = find_targets(model, cfg)
    if not targets:
        raise RuntimeError("no target layers matched cfg")

    # State buffers live on the *block*, so we need block refs even when hooking
    # sub-modules. Re-resolve blocks alongside the (possibly sub-module) target.
    from .target import _get_blocks
    blocks = _get_blocks(model)

    handles: list[RemovableHandle] = []
    attached_names: list[str] = []
    hooked_submodules: list[nn.Module] = []
    for name, mod, li in targets:
        if li not in vectors:
            raise KeyError(f"vectors missing layer {li}; have {sorted(vectors)}")
        block = blocks[li]
        _install_state(block, vectors[li], cfg)
        block._steering_cfg = cfg
        block._steering_method = method
        block._steering_layer_idx = li
        if requires_linear:
            mod._steering_block_ref = block
            hooked_submodules.append(mod)
            handles.append(mod.register_forward_hook(_linear_hook))
        else:
            handles.append(block.register_forward_hook(_hook))
        attached_names.append(name)

    setattr(model, _ATTACHED_ATTR, {
        "cfg": cfg, "targets": attached_names, "handles": handles,
        "hooked_submodules": hooked_submodules,
    })
    return handles


def detach(model: nn.Module) -> None:
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        return
    for h in state["handles"]:
        h.remove()
    for mod in state.get("hooked_submodules", []):
        if hasattr(mod, "_steering_block_ref"):
            delattr(mod, "_steering_block_ref")
    for _, block in model.named_modules():
        if not hasattr(block, "_steering_method"):
            continue
        for k in [k for k in list(block._buffers) if k.startswith(_STATE_PREFIX)]:
            del block._buffers[k]
        for attr in ("_steering_cfg", "_steering_method", "_steering_layer_idx"):
            if hasattr(block, attr):
                delattr(block, attr)
    delattr(model, _ATTACHED_ATTR)


def _log_extract_demo(tok, pos_prompts: list[str], neg_prompts: list[str]) -> None:
    """One trace per stage: decoded full prompt + special tokens, for format debugging."""
    from loguru import logger
    pos = pos_prompts[0]
    neg = neg_prompts[0]
    logger.info(
        "EXPECT: POS and NEG share user_msg + suffix; differ only in system persona; "
        "chat template applied; special tokens (e.g. <|im_start|>) visible.\n"
        "=== EXTRACT demo trace ===\n"
        f"POS[0]:\n{pos}\n---\nNEG[0]:\n{neg}\n=== /EXTRACT ==="
    )


def train(
    model: nn.Module,
    tok,
    pos_prompts: list[str],
    neg_prompts: list[str],
    cfg: SteeringConfig,
    *,
    batch_size: int = 8,
    max_length: int = 256,
):
    """repeng-style verb: extract activations + run method.extract -> Vector.

    Variants with `requires_linear_io = True` get sub-module *inputs* (via
    `extract_linear.record_linear_inputs`) and receive `model` + per-layer
    sub-modules as keyword args to `extract` so they can SVD weights.
    """
    from .vector import Vector
    _log_extract_demo(tok, pos_prompts, neg_prompts)
    method = REGISTRY[cfg.method]
    targets = find_targets(model, cfg)
    if getattr(method, "requires_linear_io", False):
        from .extract_linear import record_linear_outputs
        layer_to_module = {li: mod for _, mod, li in targets}
        pos_acts = record_linear_outputs(model, tok, pos_prompts, layer_to_module,
                                         batch_size=batch_size, max_length=max_length)
        neg_acts = record_linear_outputs(model, tok, neg_prompts, layer_to_module,
                                         batch_size=batch_size, max_length=max_length)
        state = method.extract(pos_acts, neg_acts, cfg, layer_to_module=layer_to_module)
    else:
        layers = tuple(li for _, _, li in targets)
        pos_acts = record_activations(model, tok, pos_prompts, layers, batch_size=batch_size, max_length=max_length)
        neg_acts = record_activations(model, tok, neg_prompts, layers, batch_size=batch_size, max_length=max_length)
        state = method.extract(pos_acts, neg_acts, cfg)
    return Vector(cfg, state)


def train_attn(
    model: nn.Module,
    tok,
    pos_prompts: list[str],
    neg_prompts: list[str],
    cfg: SteeringConfig,
    *,
    pool: Pool = "attn_v",
    pair_agg: PairAgg = "mean",
    batch_size: int = 8,
    max_length: int = 256,
):
    """Like train(), but with a choice of token-pooling strategy.

    pool: how to aggregate prefix tokens into a single per-prompt vector before
        running cfg.method.extract. Works with any registered method
        (mean_diff, pca, sspace, ...) -- pooling is orthogonal to direction.
        - "last": equivalent to train().
        - "mean": plain non-pad mean pooling. No attention needed.
        - "attn_v": pair-aware shared-attention pooling, isolates V content
          signal. Requires output_attentions (eager attn).
        - "attn_kq": uses attention-weight difference times shared content,
          tests K/Q routing. Requires output_attentions (eager attn).
    pair_agg: only used when pool="attn_v". Controls how (pos, neg) attention
        rows are combined: mean / max / min / hdiff. See extract_attn.

    Returns the same per-layer state dict as train().
    """
    if len(pos_prompts) != len(neg_prompts):
        raise ValueError("pos and neg prompt lists must be the same length")
    method = REGISTRY[cfg.method]
    targets = find_targets(model, cfg)
    layers = tuple(li for _, _, li in targets)

    if pool == "last":
        pos_acts = record_activations(model, tok, pos_prompts, layers,
                                      batch_size=batch_size, max_length=max_length)
        neg_acts = record_activations(model, tok, neg_prompts, layers,
                                      batch_size=batch_size, max_length=max_length)
    elif pool == "mean":
        pos_acts = record_activations_mean(model, tok, pos_prompts, layers,
                                           batch_size=batch_size, max_length=max_length)
        neg_acts = record_activations_mean(model, tok, neg_prompts, layers,
                                           batch_size=batch_size, max_length=max_length)
    elif pool in ("attn_v", "attn_kq"):
        interleaved = [p for pair in zip(pos_prompts, neg_prompts) for p in pair]
        mode: Mode = "v" if pool == "attn_v" else "kq"
        acts = record_activations_attn(
            model, tok, interleaved, layers,
            mode=mode, pair_agg=pair_agg,
            batch_size=batch_size, max_length=max_length,
        )
        pos_acts = {li: t[0::2] for li, t in acts.items()}
        neg_acts = {li: t[1::2] for li, t in acts.items()}
    else:
        raise ValueError(f"unknown pool {pool!r}")

    from .vector import Vector
    return Vector(cfg, method.extract(pos_acts, neg_acts, cfg))


def save(model: nn.Module, path: str) -> None:
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        raise RuntimeError("no steering attached; call attach() first")
    sd = {}
    for name, block in model.named_modules():
        if not hasattr(block, "_steering_method"):
            continue
        li = block._steering_layer_idx
        for k, v in block._buffers.items():
            if k.startswith(_STATE_PREFIX):
                sd[f"layer{li}.{k[len(_STATE_PREFIX):]}"] = v.detach().cpu()
    metadata = {"cfg": json.dumps(state["cfg"].to_dict())}
    from safetensors.torch import save_file
    save_file(sd, path, metadata=metadata)


def load(model: nn.Module, path: str) -> list[RemovableHandle]:
    from safetensors.torch import load_file, safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    sd = load_file(path, device="cpu")
    cfg = SteeringConfig.from_dict(json.loads(metadata["cfg"]))
    vectors: dict[int, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        layer_part, _, sub = k.partition(".")
        li = int(layer_part.removeprefix("layer"))
        vectors.setdefault(li, {})[sub] = v
    return attach(model, cfg, vectors)
