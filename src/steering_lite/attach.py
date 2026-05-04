"""attach / detach / save / load. The whole runtime.

Variant protocol (uniform across both hook paths):

    apply(block, x, y, state, cfg) -> y_new

Variants return the module's NEW output. Additive variants do `return y + delta`.
Replacing variants ignore `y` and return any tensor of the same shape. Same
contract as lora-lite's `Variant.forward`.

Two hook paths, dispatched on `cfg.target_submodule`:

  - `target_submodule is None` (default): hook each transformer **block**'s
    forward output. `x = args[0]` (input residual), `y = out[0]` (output
    hidden_states). State is keyed by `int` (block layer index).
  - `target_submodule = <regex>`: hook every nn.Linear in each selected block
    whose dotted path matches the regex. `x` is the Linear's input, `y` its
    output. State is keyed by `str` (full dotted name like
    `"layers.5.mlp.down_proj"`); each hooked submodule owns its own state
    buffers and a `_steering_block_ref` for the parent block.

The two state shapes are kept distinct so block-level methods (mean_diff,
pca, ...) and submodule-level methods (sspace*) can coexist.
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
_SUB_KEY_PREFIX = "sub::"  # safetensors key prefix marking submodule-level state
_SUB_KEY_SEP = "::"        # separator between full_name and state_key


def _gather_state(mod) -> dict[str, torch.Tensor]:
    return {
        k[len(_STATE_PREFIX):]: getattr(mod, k)
        for k in dir(mod)
        if k.startswith(_STATE_PREFIX) and isinstance(getattr(mod, k, None), torch.Tensor)
    }


def _hook(block, args, out):
    cfg: SteeringConfig = block._steering_cfg
    method = block._steering_method
    state = _gather_state(block)
    x = args[0]
    if isinstance(out, tuple):
        y = out[0]
        y_new = method.apply(block, x, y, state, cfg)
        return (y_new,) + out[1:]
    return method.apply(block, x, out, state, cfg)


def _linear_hook(mod, args, out):
    """Forward hook for sub-module variants (cfg.target_submodule is set).

    `out` is a Tensor (Linear's output), not a tuple. State + cfg + method
    live on the submodule itself; `block` is reachable via _steering_block_ref
    (kept for variant signature compatibility even though sspace* don't use it).
    """
    cfg: SteeringConfig = mod._steering_cfg
    method = mod._steering_method
    state = _gather_state(mod)
    block = mod._steering_block_ref
    return method.apply(block, args[0], out, state, cfg)


def _install_state(mod: nn.Module, state: dict[str, torch.Tensor], cfg: SteeringConfig) -> None:
    for k, v in state.items():
        attr = _STATE_PREFIX + k
        if hasattr(mod, attr):
            raise RuntimeError(f"module already has {attr}; detach first")
        mod.register_buffer(attr, v.to(cfg.dtype), persistent=True)


def attach(
    model: nn.Module,
    cfg: SteeringConfig,
    vectors,
) -> list[RemovableHandle]:
    """Install per-target state as buffers and register forward hooks.

    `vectors` shape depends on cfg.target_submodule:
      - None: dict[int, dict[str, Tensor]] keyed by block layer index.
      - regex set: dict[str, dict[str, Tensor]] keyed by full dotted name.

    Accepts a `Vector` (auto-unwrapped to its `.state`).
    """
    from .vector import Vector
    if isinstance(vectors, Vector):
        vectors = vectors.state
    if cfg.method not in REGISTRY:
        raise KeyError(f"unknown method {cfg.method!r}; registered: {list(REGISTRY)}")
    method = REGISTRY[cfg.method]
    # variant-level default target_submodule (e.g. sspace -> residual writers)
    if cfg.target_submodule is None and getattr(method, "default_target_submodule", None):
        cfg.target_submodule = method.default_target_submodule
    requires_linear = cfg.target_submodule is not None
    targets = find_targets(model, cfg)
    if not targets:
        raise RuntimeError("no target layers matched cfg")

    from .target import _get_blocks
    blocks = _get_blocks(model)

    handles: list[RemovableHandle] = []
    attached_names: list[str] = []
    hooked_modules: list[nn.Module] = []
    for full_name, mod, li in targets:
        key = full_name if requires_linear else li
        if key not in vectors:
            raise KeyError(f"vectors missing key {key!r}; have {sorted(vectors)}")
        _install_state(mod, vectors[key], cfg)
        mod._steering_cfg = cfg
        mod._steering_method = method
        if requires_linear:
            mod._steering_module_name = full_name
            mod._steering_block_ref = blocks[li]
            hooked_modules.append(mod)
            handles.append(mod.register_forward_hook(_linear_hook))
        else:
            mod._steering_layer_idx = li
            handles.append(mod.register_forward_hook(_hook))
        attached_names.append(full_name)

    setattr(model, _ATTACHED_ATTR, {
        "cfg": cfg, "targets": attached_names, "handles": handles,
        "hooked_modules": hooked_modules,
    })
    return handles


def detach(model: nn.Module) -> None:
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        return
    for h in state["handles"]:
        h.remove()
    for _, mod in model.named_modules():
        if not hasattr(mod, "_steering_method"):
            continue
        for k in [k for k in list(mod._buffers) if k.startswith(_STATE_PREFIX)]:
            del mod._buffers[k]
        for attr in (
            "_steering_cfg", "_steering_method",
            "_steering_layer_idx", "_steering_module_name", "_steering_block_ref",
        ):
            if hasattr(mod, attr):
                delattr(mod, attr)
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

    When `cfg.target_submodule` is set, recording uses
    `extract_linear.record_linear_outputs` (capturing each matched submodule's
    output) and `extract` receives a `name_to_module` kwarg so it can SVD weights.
    """
    from .vector import Vector
    _log_extract_demo(tok, pos_prompts, neg_prompts)
    method = REGISTRY[cfg.method]
    if cfg.target_submodule is None and getattr(method, "default_target_submodule", None):
        cfg.target_submodule = method.default_target_submodule
    targets = find_targets(model, cfg)
    if cfg.target_submodule is not None:
        from .extract_linear import record_linear_outputs
        name_to_module = {full_name: mod for full_name, mod, _ in targets}
        pos_acts = record_linear_outputs(model, tok, pos_prompts, name_to_module,
                                         batch_size=batch_size, max_length=max_length)
        neg_acts = record_linear_outputs(model, tok, neg_prompts, name_to_module,
                                         batch_size=batch_size, max_length=max_length)
        state = method.extract(pos_acts, neg_acts, cfg, name_to_module=name_to_module)
    else:
        layers = tuple(li for _, _, li in targets)
        pos_acts = record_activations(model, tok, pos_prompts, layers, batch_size=batch_size, max_length=max_length)
        neg_acts = record_activations(model, tok, neg_prompts, layers, batch_size=batch_size, max_length=max_length)
        if getattr(method, "needs_model", False):
            state = method.extract(pos_acts, neg_acts, cfg, model=model)
        else:
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


def _state_to_safetensors_dict(model: nn.Module) -> dict:
    """Serialise installed state buffers from all hooked modules. Forks on
    whether the module is block-level (_steering_layer_idx) or submodule-level
    (_steering_module_name); keys distinguish the two so load() can rebuild."""
    sd = {}
    for _, mod in model.named_modules():
        if not hasattr(mod, "_steering_method"):
            continue
        if hasattr(mod, "_steering_module_name"):
            full_name = mod._steering_module_name
            for k, v in mod._buffers.items():
                if k.startswith(_STATE_PREFIX):
                    sd[f"{_SUB_KEY_PREFIX}{full_name}{_SUB_KEY_SEP}{k[len(_STATE_PREFIX):]}"] = v.detach().cpu()
        elif hasattr(mod, "_steering_layer_idx"):
            li = mod._steering_layer_idx
            for k, v in mod._buffers.items():
                if k.startswith(_STATE_PREFIX):
                    sd[f"layer{li}.{k[len(_STATE_PREFIX):]}"] = v.detach().cpu()
    return sd


def _safetensors_dict_to_state(sd: dict[str, torch.Tensor]) -> dict:
    """Inverse of _state_to_safetensors_dict. Returns dict keyed by int (block-level)
    or str (submodule-level), depending on the prefix of each key."""
    vectors: dict = {}
    for k, v in sd.items():
        if k.startswith(_SUB_KEY_PREFIX):
            rest = k[len(_SUB_KEY_PREFIX):]
            full_name, _, state_key = rest.rpartition(_SUB_KEY_SEP)
            vectors.setdefault(full_name, {})[state_key] = v
        else:
            layer_part, _, sub = k.partition(".")
            li = int(layer_part.removeprefix("layer"))
            vectors.setdefault(li, {})[sub] = v
    return vectors


def save(model: nn.Module, path: str) -> None:
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        raise RuntimeError("no steering attached; call attach() first")
    sd = _state_to_safetensors_dict(model)
    metadata = {"cfg": json.dumps(state["cfg"].to_dict())}
    from safetensors.torch import save_file
    save_file(sd, path, metadata=metadata)


def load(model: nn.Module, path: str) -> list[RemovableHandle]:
    from safetensors.torch import load_file, safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    sd = load_file(path, device="cpu")
    cfg = SteeringConfig.from_dict(json.loads(metadata["cfg"]))
    vectors = _safetensors_dict_to_state(sd)
    return attach(model, cfg, vectors)
