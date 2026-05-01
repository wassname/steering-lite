"""SteeringConfig: per-method typed dataclass.

Each method ships its own subclass under `variants/*.py` (e.g. `MeanDiffC`),
adding strongly-typed knobs. Registry route at load time: `from_dict` looks up the
right subclass via the `method` field.
"""
from dataclasses import dataclass, asdict, field
import torch


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
