"""Vector: extracted steering vector + config, as a single ergonomic object.

Wraps `(cfg, state)` so a user can:

    v = Vector.train(model, tok, pos, neg, sl.MeanDiffC(layers=(15,))) \\
              .calibrate(model, tok, target_kl=1.0)

    with v(model):
        out = model.generate(...)

    v.save("honesty.safetensors")
    v2 = Vector.load("honesty.safetensors")

    combined = v + v2          # ensemble (sum buffers, requires same cfg.method)
    scaled   = v * 0.5         # scale buffers
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import torch
from torch import Tensor, nn

from .config import SteeringConfig


class Vector:
    def __init__(self, cfg: SteeringConfig, state: dict[int, dict[str, Tensor]]):
        self.cfg = cfg
        self.state = state

    @classmethod
    def train(cls, model: nn.Module, tok, pos_prompts: list[str], neg_prompts: list[str],
              cfg: SteeringConfig, **kw) -> "Vector":
        """Extract a steering vector from contrastive prompts. Chains with .calibrate()."""
        from .attach import train as _train
        return _train(model, tok, pos_prompts, neg_prompts, cfg, **kw)

    def calibrate(self, model: nn.Module, tok,
                  prompts: list[str] | list[Tensor] | None = None, *,
                  target_kl: float = 1.0, **kw) -> "Vector":
        """Set coeff so KL(steered || base) hits target_kl. Mutates and returns self for chaining.

        `prompts` defaults to a small generic set; pass list[str] to use your own.
        """
        from .calibrate import calibrate_iso_kl
        coeff, _ = calibrate_iso_kl(self, model, tok, prompts, target_kl=target_kl, **kw)
        self.cfg.coeff = float(coeff)
        return self

    @contextmanager
    def __call__(self, model: nn.Module, *, C: float | None = None):
        """Attach for the duration of the `with` block. `C` overrides cfg.coeff if given."""
        from .attach import attach, detach
        cfg = self.cfg if C is None else replace(self.cfg, coeff=float(C))
        attach(model, cfg, self.state)
        try:
            yield model
        finally:
            detach(model)

    def __add__(self, other: "Vector") -> "Vector":
        if self.cfg.method != other.cfg.method:
            raise ValueError(f"cannot add {self.cfg.method!r} + {other.cfg.method!r}")
        if sorted(self.state) != sorted(other.state):
            raise ValueError(f"layer mismatch: {sorted(self.state)} vs {sorted(other.state)}")
        new_state: dict[int, dict[str, Tensor]] = {}
        for li in self.state:
            a, b = self.state[li], other.state[li]
            if sorted(a) != sorted(b):
                raise ValueError(f"layer {li}: state keys differ {sorted(a)} vs {sorted(b)}")
            new_state[li] = {k: a[k] + b[k] for k in a}
        return Vector(deepcopy(self.cfg), new_state)

    def __mul__(self, k: float) -> "Vector":
        new_state = {
            li: {k_: v * float(k) for k_, v in s.items()}
            for li, s in self.state.items()
        }
        return Vector(deepcopy(self.cfg), new_state)

    __rmul__ = __mul__

    def save(self, path: str) -> None:
        from .attach import _STATE_PREFIX, _SUB_KEY_PREFIX, _SUB_KEY_SEP  # noqa: F401
        import json
        from safetensors.torch import save_file
        sd: dict[str, Tensor] = {}
        sub_mode = self.cfg.target_submodule is not None
        for key, s in self.state.items():
            for k, t in s.items():
                if sub_mode:
                    sd[f"{_SUB_KEY_PREFIX}{key}{_SUB_KEY_SEP}{k}"] = t.detach().cpu()
                else:
                    sd[f"layer{key}.{k}"] = t.detach().cpu()
        metadata = {"cfg": json.dumps(self.cfg.to_dict())}
        save_file(sd, path, metadata=metadata)

    @classmethod
    def load(cls, path: str) -> "Vector":
        import json
        from safetensors.torch import load_file, safe_open
        from .attach import _safetensors_dict_to_state
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
        sd = load_file(path, device="cpu")
        cfg = SteeringConfig.from_dict(json.loads(metadata["cfg"]))
        state = _safetensors_dict_to_state(sd)
        return cls(cfg, state)

    def __repr__(self) -> str:
        layers = sorted(self.state)
        return f"Vector(method={self.cfg.method!r}, layers={layers})"
