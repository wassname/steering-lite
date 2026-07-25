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


@torch.no_grad()
def collect_pairs(
    model: nn.Module,
    tok,
    texts: list[str],
    layers: tuple[int, ...],
    *,
    batch_size: int = 8,
    max_length: int = 128,
) -> tuple[dict[int, Tensor], Tensor]:
    """Residuals at each layer and at the end, one row per real token position."""
    device = next(model.parameters()).device
    per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    finals: list[Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        hidden = model(**enc, output_hidden_states=True).hidden_states
        keep = enc["attention_mask"].bool().flatten()
        for layer in layers:
            per_layer[layer].append(hidden[layer].flatten(0, 1)[keep].float().cpu())
        finals.append(hidden[-1].flatten(0, 1)[keep].float().cpu())
    return {l: torch.cat(rows) for l, rows in per_layer.items()}, torch.cat(finals)


def fit_ridge(
    X: Float[Tensor, "n d"], Y: Float[Tensor, "n d"], ridge: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Least-squares A, b with Y ~ X A^T + b, solved on the centred covariance."""
    x_mean, y_mean = X.mean(0), Y.mean(0)
    Xc, Yc = X - x_mean, Y - y_mean
    gram = Xc.T @ Xc
    gram.diagonal().add_(ridge * gram.diagonal().mean())
    A = torch.linalg.solve(gram, Xc.T @ Yc).T
    return A, y_mean - A @ x_mean


class TunedLens:
    """Per-layer affine maps into the final residual, plus the model's own unembedding head."""

    def __init__(self, translators: dict[int, tuple[Tensor, Tensor]]):
        self.translators = translators

    @classmethod
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
    ) -> "TunedLens":
        per_layer, finals = collect_pairs(
            model, tok, texts, layers, batch_size=batch_size, max_length=max_length
        )
        translators = {}
        logger.info(
            "SHOULD: r2 rises with depth and is well above 0 at every layer, since a later residual "
            "predicts the final one better. ELSE the fit is rank-starved and the readout is noise."
        )
        for layer in layers:
            X, Y = per_layer[layer], finals
            A, b = fit_ridge(X, Y, ridge=ridge)
            residual = Y - (X @ A.T + b)
            r2 = 1.0 - residual.var(0).sum().item() / Y.var(0).sum().item()
            logger.info(f"  layer {layer:>3}  n={len(X):>7}  r2={r2:+.3f}")
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
