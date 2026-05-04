"""SteeringConfig + Method protocol + registries.

Each method ships its own subclass `XC(SteeringConfig)` and `XMethod` class
under `variants/*.py` (e.g. `MeanDiffC` + `MeanDiff`). Two parallel registries
keyed by method name: `_CONFIG_REGISTRY` for `from_dict` deserialisation,
`REGISTRY` for the runtime extract/apply pair.
"""
from dataclasses import dataclass, asdict, field
from typing import Protocol, Any
import torch
from torch import Tensor


@dataclass
class SteeringConfig:
    """Base config. Subclass per method; do not instantiate directly."""
    method: str = "?"

    # which transformer blocks to hook (indices into model.model.layers)
    # None = all layers
    layers: tuple[int, ...] | None = None

    # which point in the block to add at: "residual" = block output (post mlp+attn),
    # "attn_out" = attention output, "mlp_out" = mlp output.
    # v1 only implements "residual".
    target: str = "residual"

    # Optional dotted path of a sub-module within each target block to hook on
    # (e.g. "mlp.down_proj"). When None, the block's forward output is hooked
    # (default for almost all variants); when set, the sub-module's forward is
    # hooked instead. Either way the variant's apply receives the unified
    # (block, x, y, state, cfg) signature -- used by weight-SVD methods
    # (sspace, sspace_ablate) that need to modify a Linear's output in low-rank
    # S-space.
    target_submodule: str | None = None

    # steering strength at apply-time. Methods interpret it differently:
    # additive (mean_diff, pca, sspace, chars, cosine_gated): coeff is α in `h + α v`.
    # slerp/angle (spherical, angular_steering): coeff is the slerp t / rotation θ.
    # blend (linear_act): coeff is the blend ratio.
    # ablation+nudge (directional_ablation): coeff is post-ablation nudge magnitude.
    coeff: float = 1.0

    dtype: torch.dtype = torch.bfloat16
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dtype"] = str(self.dtype).removeprefix("torch.")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringConfig":
        d = dict(d)
        name = d["method"]
        sub = _CONFIG_REGISTRY[name]
        d["dtype"] = getattr(torch, d["dtype"])
        return sub(**d)


_CONFIG_REGISTRY: dict[str, type[SteeringConfig]] = {}


def register_config(cls: type[SteeringConfig]) -> type[SteeringConfig]:
    """Decorator: register `cls` under its `method` default value."""
    name = cls.__dataclass_fields__["method"].default
    if name == "?":
        raise ValueError(f"{cls} must override the default `method` field")
    if name in _CONFIG_REGISTRY:
        raise ValueError(f"config for method {name!r} already registered")
    _CONFIG_REGISTRY[name] = cls
    return cls


class Method(Protocol):
    """extract+apply pair. State tensors are registered as buffers on the hooked
    module (block or Linear) under `_steering_state_<key>` and rebuilt into a
    dict by the hook.
    """
    name: str

    @staticmethod
    def extract(
        pos_acts: dict[int, Tensor],
        neg_acts: dict[int, Tensor],
        cfg: Any,
    ) -> dict[int, dict[str, Tensor]]:
        """Per-layer state. `pos_acts[l]` is `[n_pos, d_model]`, same for neg."""
        ...

    @staticmethod
    def apply(
        mod,           # the hooked module: a transformer block, or a Linear
        x: Tensor,     # [b, s, d_in]  -- module input
        y: Tensor,     # [b, s, d_out] -- module output
        state: dict[str, Tensor],
        cfg: Any,
    ) -> Tensor:
        """Return the module's NEW output. Additive variants: `return y + delta`.
        Replacing variants: ignore `y`, return any tensor of shape `[b, s, d_out]`.
        """
        ...


REGISTRY: dict[str, type] = {}


def register(cls):
    if not getattr(cls, "name", None):
        raise ValueError(f"method {cls} missing .name")
    REGISTRY[cls.name] = cls
    return cls
