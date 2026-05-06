"""Vector: extracted steering vector + config, with shared / stacked state split.

`shared` holds module-tied tensors (basis, biases) that are properties of the
weight matrix and don't accumulate. `stacked` holds the contrastive directions
extracted from POS/NEG pairs, with a leading k-dim that grows by 1 each time
two Vectors are added. Each variant decides which keys go where.

`Vector + Vector` natural-fail design:
  - shared keys are checked with `allclose`; same-basis methods (sspace etc)
    pass, while methods that put their per-contrast direction in shared
    (cosine_gated, pca, ...) fail because contrasts differ across rounds.
  - stacked tensors are cat'd along dim 0 (k_a + k_b rounds).
  - methods opt in to multi-round simply by placing per-contrast tensors in
    `stacked` and writing apply() to handle the leading k-dim.

`Vector * alpha` scales every stacked tensor uniformly. Magnitudes of
stacked rows carry per-direction calibration -- e.g. for sspace, row norm
of dS = alpha_i, direction = dS / row_norm. apply() normalizes on the fly
so an outer * alpha just scales all per-direction calibrations linearly.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import torch
from torch import Tensor, nn

from .config import SteeringConfig


def _allclose_tree(a: dict, b: dict, *, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    if sorted(a) != sorted(b):
        raise ValueError(f"shared keys differ: {sorted(a)} vs {sorted(b)}")
    for k in a:
        if a[k].shape != b[k].shape:
            raise ValueError(f"shared[{k!r}] shape differs: {a[k].shape} vs {b[k].shape}")
        if not torch.allclose(a[k].float(), b[k].float(), atol=atol, rtol=rtol):
            raise ValueError(f"shared[{k!r}] tensors differ; basis must match across rounds")


class Vector:
    def __init__(
        self,
        cfg: SteeringConfig,
        shared:  dict,    # {layer_key: {name: Tensor}}
        stacked: dict,    # {layer_key: {name: Tensor with leading k-dim}}
    ):
        self.cfg = cfg
        self.shared = shared
        self.stacked = stacked

    @classmethod
    def train(cls, model: nn.Module, tok, pos_prompts: list[str], neg_prompts: list[str],
              cfg: SteeringConfig, **kw) -> "Vector":
        from .attach import train as _train
        return _train(model, tok, pos_prompts, neg_prompts, cfg, **kw)

    def calibrate(self, model: nn.Module, tok,
                  prompts: list[str] | list[Tensor] | None = None, *,
                  target_kl: float = 1.0, **kw) -> "Vector":
        from .calibrate import calibrate_iso_kl
        coeff, _ = calibrate_iso_kl(self, model, tok, prompts, target_kl=target_kl, **kw)
        self.cfg.coeff = float(coeff)
        return self

    @contextmanager
    def __call__(self, model: nn.Module, *, C: float | None = None):
        from .attach import attach, detach
        cfg = self.cfg if C is None else replace(self.cfg, coeff=float(C))
        attach(model, cfg, self.shared, self.stacked)
        try:
            yield model
        finally:
            detach(model)

    def __add__(self, other: "Vector") -> "Vector":
        if self.cfg.method != other.cfg.method:
            raise ValueError(f"cannot add {self.cfg.method!r} + {other.cfg.method!r}")
        if self.cfg.target_submodule != other.cfg.target_submodule:
            raise ValueError(
                f"target_submodule mismatch: {self.cfg.target_submodule!r} vs "
                f"{other.cfg.target_submodule!r}")
        if sorted(self.shared) != sorted(other.shared):
            raise ValueError(f"layer keys differ: {sorted(self.shared)} vs {sorted(other.shared)}")
        new_shared, new_stacked = {}, {}
        for li in self.shared:
            _allclose_tree(self.shared[li], other.shared[li])
            new_shared[li] = {k: v.clone() for k, v in self.shared[li].items()}
            a, b = self.stacked[li], other.stacked[li]
            if sorted(a) != sorted(b):
                raise ValueError(f"layer {li}: stacked keys differ {sorted(a)} vs {sorted(b)}")
            new_stacked[li] = {k: torch.cat([a[k], b[k]], dim=0) for k in a}
        return Vector(deepcopy(self.cfg), new_shared, new_stacked)

    def __mul__(self, k: float) -> "Vector":
        new_stacked = {
            li: {key: t * float(k) for key, t in s.items()}
            for li, s in self.stacked.items()
        }
        new_shared = {
            li: {key: t.clone() for key, t in s.items()}
            for li, s in self.shared.items()
        }
        return Vector(deepcopy(self.cfg), new_shared, new_stacked)

    __rmul__ = __mul__

    def k_rounds(self) -> int:
        """How many directions have been accumulated (leading dim of stacked)."""
        for li in self.stacked:
            for _, t in self.stacked[li].items():
                return int(t.shape[0])
        return 0

    def save(self, path: str) -> None:
        from .attach import _SUB_KEY_PREFIX, _SUB_KEY_SEP
        import json
        from safetensors.torch import save_file
        sd: dict[str, Tensor] = {}
        sub_mode = self.cfg.target_submodule is not None
        for kind, tree in (("shared", self.shared), ("stacked", self.stacked)):
            for layer_key, s in tree.items():
                for k, t in s.items():
                    if sub_mode:
                        sd[f"{_SUB_KEY_PREFIX}{kind}::{layer_key}{_SUB_KEY_SEP}{k}"] = t.detach().cpu()
                    else:
                        sd[f"{kind}.layer{layer_key}.{k}"] = t.detach().cpu()
        metadata = {"cfg": json.dumps(self.cfg.to_dict())}
        save_file(sd, path, metadata=metadata)

    @classmethod
    def load(cls, path: str) -> "Vector":
        import json
        from safetensors.torch import load_file, safe_open
        from .attach import _safetensors_dict_to_split_state
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
        sd = load_file(path, device="cpu")
        cfg = SteeringConfig.from_dict(json.loads(metadata["cfg"]))
        shared, stacked = _safetensors_dict_to_split_state(sd)
        return cls(cfg, shared, stacked)

    def __repr__(self) -> str:
        layers = sorted(self.shared) if self.shared else sorted(self.stacked)
        return (f"Vector(method={self.cfg.method!r}, layers={layers}, "
                f"k_rounds={self.k_rounds()})")
