import os as _os

if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import SteeringConfig, REGISTRY, register
from .extract import record_activations
from .extract_attn import record_activations_attn
from .attach import attach, detach, save, load, train, train_attn
from .calibrate import measure_kl, calibrate_iso_kl
from . import variants  # noqa: F401  triggers method + config registration
from .vector import Vector

from .variants.mean_diff import MeanDiffC
from .variants.pca import PCAC
from .variants.topk_clusters import TopKClustersC
from .variants.cosine_gated import CosineGatedC
from .variants.sspace import SSpaceC
from .variants.spherical import SphericalC
from .variants.directional_ablation import DirectionalAblationC
from .variants.chars import CHaRSC
from .variants.linear_act import LinearAcTC
from .variants.angular_steering import AngularSteeringC

__all__ = [
    "SteeringConfig",
    "MeanDiffC",
    "PCAC",
    "TopKClustersC",
    "CosineGatedC",
    "SSpaceC",
    "SphericalC",
    "DirectionalAblationC",
    "CHaRSC",
    "LinearAcTC",
    "AngularSteeringC",
    "record_activations",
    "record_activations_attn",
    "train",
    "train_attn",
    "attach",
    "detach",
    "save",
    "load",
    "measure_kl",
    "calibrate_iso_kl",
    "REGISTRY",
    "register",
    "Vector",
]
