"""Mean-difference steering (CAA / ActAdd).

For each selected layer L, compute the mean difference between positive and
negative last-token hidden states:

$$v_L = \\text{mean}(h^+_L) - \\text{mean}(h^-_L), \\quad \\hat{v}_L = v_L / \\|v_L\\|$$

At runtime, add `coeff * v_hat` to every token's residual at that block:

$$h \\leftarrow h + \\alpha \\cdot \\hat{v}_L$$

Refs:
  - Panickssery et al. 2023 https://arxiv.org/abs/2312.06681 (CAA)
  - Turner et al. 2023 https://arxiv.org/abs/2308.10248 (ActAdd)
  - nrimsky/CAA: https://github.com/nrimsky/CAA
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config
from ..method import register


@register_config
@dataclass
class MeanDiffConfig(SteeringConfig):
    method: str = "mean_diff"
    normalize: bool = True
    subtract_corpus_mean: bool = False


@register
class MeanDiff:
    name = "mean_diff"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "m d"]],
        cfg: MeanDiffConfig,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            p = pos_acts[li].float()
            n = neg_acts[li].float()
            if cfg.subtract_corpus_mean:
                # Jorgensen 2024 mean-centring: target mean minus training/corpus mean.
                # Under this paired API the available corpus is pos∪neg, so equal groups
                # give 0.5 * mean_diff; with normalization it is direction-identical.
                mu = torch.cat([p, n], dim=0).mean(0)
                v = p.mean(0) - mu
            else:
                v = p.mean(0) - n.mean(0)
            if cfg.normalize:
                v = v / v.norm()
            out[li] = {"v": v}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: MeanDiffConfig,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(h.dtype).to(h.device)
        return h + cfg.coeff * v


# CAA (Panickssery 2023) and ActAdd (Turner 2023) are the same operation as
# mean_diff -- the difference is conventional, not mathematical:
#   - CAA: many contrastive MCQ pairs, last token of each
#   - ActAdd: one prompt-pair difference at a chosen layer/token
#   - mean_diff: same math, register both as aliases for drop-in benchmarking
# Aliases subclass the config (so cfg.method round-trips) and re-register the
# class under a new .name. extract/apply are inherited unchanged.

@register_config
@dataclass
class CAAConfig(MeanDiffConfig):
    method: str = "caa"


@register
class CAA(MeanDiff):
    name = "caa"


@register_config
@dataclass
class ActAddConfig(MeanDiffConfig):
    method: str = "act_add"


@register
class ActAdd(MeanDiff):
    name = "act_add"


# Mean-Centring (Jorgensen 2024 https://arxiv.org/abs/2312.03813). In this
# paired API it uses pos∪neg as the corpus baseline; use a separate corpus
# extractor later if we need a paper-faithful unpaired training distribution.

@register_config
@dataclass
class MeanCentredConfig(MeanDiffConfig):
    method: str = "mean_centred"
    subtract_corpus_mean: bool = True


@register
class MeanCentred(MeanDiff):
    name = "mean_centred"
