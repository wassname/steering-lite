"""Fit a translator from a mid-layer residual to the final residual, so directions can be read.

The plain logit lens (`final_norm(v) @ W_U.T`) does not work at the depths steering vectors are
built at. Measured on Qwen2.5-7B: the real hidden state for a prompt whose answer is ' Tokyo' lenses
to Chinese boilerplate at layers 10-15 and only resolves at layer 25. A readout taken there says
nothing about the vector.

The fix is the tuned lens (Belrose et al. 2303.08112) and the jacobian-lens family: learn an affine
map A_L, b_L with h_final ~ A_L h_L + b_L, then read a direction as W_U · final_norm(A_L v). The
corpus is spent once at fit time, so the readout itself takes no prompt, no stem and no coefficient,
which is the property that makes it safe to point at any vector.

Fit is closed-form ridge regression. Every token position is a sample, so a few hundred sequences
give ~100k samples for a d-dimensional fit, comfortably overdetermined.
"""
from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float
from loguru import logger
from torch import Tensor, nn


class RidgeAccumulator:
    """Sufficient statistics for Y ~ X A^T + b, so nothing has to hold the activations.

    Storing them is what makes this run out of memory: 6 layers of 65k tokens at d=3584 is 5.6GB,
    doubled by the concatenate. The normal equations only need X^T X and X^T Y, which are d x d
    regardless of how many tokens go through.
    """

    def __init__(self, d: int, device: str = "cpu"):
        self.xtx = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.xty = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.sum_x = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_y = torch.zeros(d, dtype=torch.float64, device=device)
        self.n = 0

    def update(self, X: Float[Tensor, "n d"], Y: Float[Tensor, "n d"]) -> None:
        X, Y = X.to(self.xtx.dtype), Y.to(self.xtx.dtype)
        self.xtx += X.T @ X
        self.xty += X.T @ Y
        self.sum_x += X.sum(0)
        self.sum_y += Y.sum(0)
        self.n += X.shape[0]

    def solve(self, ridge: float = 1.0) -> tuple[Tensor, Tensor]:
        if self.n == 0:
            raise ValueError("no rows reached the fit; holdout_rows swallowed the whole corpus")
        mean_x, mean_y = self.sum_x / self.n, self.sum_y / self.n
        gram = self.xtx - self.n * torch.outer(mean_x, mean_x)
        cross = self.xty - self.n * torch.outer(mean_x, mean_y)
        gram.diagonal().add_(ridge * gram.diagonal().mean())
        A = torch.linalg.solve(gram, cross).T
        return A.float().cpu(), (mean_y - A @ mean_x).float().cpu()


class TunedLens:
    """Per-layer affine maps into the final residual, plus the model's own unembedding head."""

    def __init__(self, translators: dict[int, tuple[Tensor, Tensor]]):
        self.translators = translators

    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        model: nn.Module,
        tok,
        texts: list[str],
        layers: tuple[int, ...],
        *,
        batch_size: int = 8,
        max_length: int = 128,
        ridge: float = 1.0,
        holdout_rows: int = 4096,
    ) -> "TunedLens":
        device = next(model.parameters()).device
        d_model = model.config.hidden_size
        accumulators = {layer: RidgeAccumulator(d_model, device=str(device)) for layer in layers}
        held: dict[int, list[Tensor]] = {layer: [] for layer in layers}
        held_final: list[Tensor] = []
        held_rows = 0

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tok(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            hidden = model(**enc, output_hidden_states=True).hidden_states
            keep = enc["attention_mask"].bool().flatten()
            final = hidden[-1].flatten(0, 1)[keep].float()
            # The first batches are held out, so r2 is measured on rows the fit never saw.
            to_holdout = held_rows < holdout_rows
            if to_holdout:
                held_final.append(final.cpu())
                held_rows += final.shape[0]
            for layer in layers:
                rows = hidden[layer].flatten(0, 1)[keep].float()
                if to_holdout:
                    held[layer].append(rows.cpu())
                else:
                    accumulators[layer].update(rows, final)

        translators = {}
        logger.info(
            "SHOULD: held-out r2 is well above 0 at every layer and rises with depth, since a later "
            "residual predicts the final one better. ELSE the fit is rank-starved and reads as noise."
        )
        Y_held = torch.cat(held_final)
        for layer in layers:
            A, b = accumulators[layer].solve(ridge=ridge)
            X_held = torch.cat(held[layer])
            residual = Y_held - (X_held @ A.T + b)
            r2 = 1.0 - residual.var(0).sum().item() / Y_held.var(0).sum().item()
            logger.info(
                f"  layer {layer:>3}  n_fit={accumulators[layer].n:>7}  "
                f"n_held={len(X_held):>6}  r2={r2:+.3f}"
            )
            translators[layer] = (A, b)
        return cls(translators)

    def to_final(self, direction: Float[Tensor, "d"], layer: int) -> Float[Tensor, "d"]:
        """A direction is a difference of hidden states, so only the linear part carries over."""
        A, _ = self.translators[layer]
        return A.to(direction.device, direction.dtype) @ direction

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        flat = {}
        for layer, (A, b) in self.translators.items():
            flat[f"{layer}.A"], flat[f"{layer}.b"] = A.cpu().clone(), b.cpu().clone()
        save_file(flat, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "TunedLens":
        from safetensors.torch import load_file

        flat = load_file(str(path))
        layers = sorted({int(key.split(".")[0]) for key in flat})
        return cls({layer: (flat[f"{layer}.A"], flat[f"{layer}.b"]) for layer in layers})
