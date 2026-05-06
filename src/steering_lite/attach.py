"""attach / detach / save / load. The whole runtime.

Variant protocol (uniform across both hook paths):

    apply(mod, x, y, shared, stacked, cfg) -> y_new

`mod` is the hooked module itself (a transformer block or a Linear); `x` is
its input, `y` its output. `shared` holds singleton tensors per module
(e.g. SVD basis, biases). `stacked` holds tensors with a leading k-dim
(one slice per accumulated round). Variants return the module's NEW output.

Two hook paths, dispatched on `cfg.target_submodule`:

  - `target_submodule is None` (default): hook each transformer block's
    forward output. State keyed by `int` (block index).
  - `target_submodule = <regex>`: hook every nn.Linear in each selected block
    whose dotted path matches. State keyed by `str` (full dotted name).
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
_SHARED_PREFIX  = "_steering_shared_"
_STACKED_PREFIX = "_steering_stacked_"
_SUB_KEY_PREFIX = "sub::"
_SUB_KEY_SEP = "::"


def _gather_split_state(mod) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    shared, stacked = {}, {}
    for k in dir(mod):
        if k.startswith(_SHARED_PREFIX):
            t = getattr(mod, k, None)
            if isinstance(t, torch.Tensor):
                shared[k[len(_SHARED_PREFIX):]] = t
        elif k.startswith(_STACKED_PREFIX):
            t = getattr(mod, k, None)
            if isinstance(t, torch.Tensor):
                stacked[k[len(_STACKED_PREFIX):]] = t
    return shared, stacked


def _hook(mod, args, out):
    cfg: SteeringConfig = mod._steering_cfg
    method = mod._steering_method
    shared, stacked = _gather_split_state(mod)
    x = args[0]
    if isinstance(out, tuple):
        y = out[0]
        y_new = method.apply(mod, x, y, shared, stacked, cfg)
        return (y_new,) + out[1:]
    return method.apply(mod, x, out, shared, stacked, cfg)


def _linear_hook(mod, args, out):
    cfg: SteeringConfig = mod._steering_cfg
    method = mod._steering_method
    shared, stacked = _gather_split_state(mod)
    return method.apply(mod, args[0], out, shared, stacked, cfg)


def _install_state(mod: nn.Module, shared: dict, stacked: dict, cfg: SteeringConfig) -> None:
    for k, v in shared.items():
        attr = _SHARED_PREFIX + k
        if hasattr(mod, attr):
            raise RuntimeError(f"module already has {attr}; detach first")
        mod.register_buffer(attr, v.to(cfg.dtype), persistent=True)
    for k, v in stacked.items():
        attr = _STACKED_PREFIX + k
        if hasattr(mod, attr):
            raise RuntimeError(f"module already has {attr}; detach first")
        mod.register_buffer(attr, v.to(cfg.dtype), persistent=True)


def attach(
    model: nn.Module,
    cfg: SteeringConfig,
    shared,
    stacked,
) -> list[RemovableHandle]:
    """Install per-target shared+stacked state and register forward hooks.

    `shared` / `stacked`: dict[layer_key, dict[str, Tensor]]. layer_key is
    int when cfg.target_submodule is None, else str (full dotted name).

    Accepts a `Vector` as the first positional arg (auto-unwraps).
    """
    from .vector import Vector
    if isinstance(shared, Vector):
        v: Vector = shared
        cfg = v.cfg
        shared = v.shared
        stacked = v.stacked
    if cfg.method not in REGISTRY:
        raise KeyError(f"unknown method {cfg.method!r}; registered: {list(REGISTRY)}")
    method = REGISTRY[cfg.method]
    if cfg.target_submodule is None and getattr(method, "default_target_submodule", None):
        cfg.target_submodule = method.default_target_submodule
    requires_linear = cfg.target_submodule is not None
    targets = find_targets(model, cfg)
    if not targets:
        raise RuntimeError("no target layers matched cfg")

    handles: list[RemovableHandle] = []
    attached_names: list[str] = []
    hooked_modules: list[nn.Module] = []
    for full_name, mod, li in targets:
        key = full_name if requires_linear else li
        sh = shared.get(key, {})
        st = stacked.get(key, {})
        if not sh and not st:
            raise KeyError(f"no state for key {key!r}; "
                           f"shared keys: {sorted(shared)}, stacked keys: {sorted(stacked)}")
        _install_state(mod, sh, st, cfg)
        mod._steering_cfg = cfg
        mod._steering_method = method
        if requires_linear:
            mod._steering_module_name = full_name
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
        for k in [k for k in list(mod._buffers)
                  if k.startswith(_SHARED_PREFIX) or k.startswith(_STACKED_PREFIX)]:
            del mod._buffers[k]
        for attr in (
            "_steering_cfg", "_steering_method",
            "_steering_layer_idx", "_steering_module_name",
        ):
            if hasattr(mod, attr):
                delattr(mod, attr)
    delattr(model, _ATTACHED_ATTR)


def _log_extract_demo(tok, pos_prompts: list[str], neg_prompts: list[str]) -> None:
    from loguru import logger
    pos = pos_prompts[0]
    neg = neg_prompts[0]
    logger.info(
        "EXPECT: POS and NEG share user_msg + suffix; differ only in system persona; "
        "chat template applied; special tokens (e.g. <|im_start|>) visible.\n"
        "=== EXTRACT demo trace ===\n"
        f"POS[0]:\n{pos}\n---\nNEG[0]:\n{neg}\n=== /EXTRACT ==="
    )


def _split_extracted(extracted: dict) -> tuple[dict, dict]:
    """Split extract() return value into (shared, stacked) dicts.

    extract() returns {layer_key: {"shared": {...}, "stacked": {...}}}.
    """
    shared, stacked = {}, {}
    for layer_key, parts in extracted.items():
        shared[layer_key] = parts.get("shared", {})
        stacked[layer_key] = parts["stacked"]
    return shared, stacked


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
        extracted = method.extract(pos_acts, neg_acts, cfg, name_to_module=name_to_module)
    else:
        layers = tuple(li for _, _, li in targets)
        pos_acts = record_activations(model, tok, pos_prompts, layers, batch_size=batch_size, max_length=max_length)
        neg_acts = record_activations(model, tok, neg_prompts, layers, batch_size=batch_size, max_length=max_length)
        if getattr(method, "needs_model", False):
            extracted = method.extract(pos_acts, neg_acts, cfg, model=model)
        else:
            extracted = method.extract(pos_acts, neg_acts, cfg)
    shared, stacked = _split_extracted(extracted)
    return Vector(cfg, shared, stacked)


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
    extracted = method.extract(pos_acts, neg_acts, cfg)
    shared, stacked = _split_extracted(extracted)
    return Vector(cfg, shared, stacked)


def _state_to_safetensors_dict(model: nn.Module) -> dict:
    sd = {}
    for _, mod in model.named_modules():
        if not hasattr(mod, "_steering_method"):
            continue
        is_sub = hasattr(mod, "_steering_module_name")
        layer_id = mod._steering_module_name if is_sub else mod._steering_layer_idx
        for k, v in mod._buffers.items():
            if k.startswith(_SHARED_PREFIX):
                kind, name = "shared", k[len(_SHARED_PREFIX):]
            elif k.startswith(_STACKED_PREFIX):
                kind, name = "stacked", k[len(_STACKED_PREFIX):]
            else:
                continue
            if is_sub:
                sd[f"{_SUB_KEY_PREFIX}{kind}::{layer_id}{_SUB_KEY_SEP}{name}"] = v.detach().cpu()
            else:
                sd[f"{kind}.layer{layer_id}.{name}"] = v.detach().cpu()
    return sd


def _safetensors_dict_to_split_state(sd: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Inverse of _state_to_safetensors_dict. Returns (shared, stacked)
    dicts, each keyed by int (block-level) or str (submodule-level)."""
    shared, stacked = {}, {}
    for k, v in sd.items():
        if k.startswith(_SUB_KEY_PREFIX):
            rest = k[len(_SUB_KEY_PREFIX):]
            kind_part, _, rest2 = rest.partition("::")
            full_name, _, state_key = rest2.rpartition(_SUB_KEY_SEP)
            target = shared if kind_part == "shared" else stacked
            target.setdefault(full_name, {})[state_key] = v
        else:
            kind_part, _, rest = k.partition(".")
            layer_part, _, state_key = rest.partition(".")
            li = int(layer_part.removeprefix("layer"))
            target = shared if kind_part == "shared" else stacked
            target.setdefault(li, {})[state_key] = v
    return shared, stacked


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
    shared, stacked = _safetensors_dict_to_split_state(sd)
    return attach(model, cfg, shared, stacked)
