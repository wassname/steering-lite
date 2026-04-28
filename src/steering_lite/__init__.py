import os as _os

if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import SteeringConfig
from .extract import record_activations
from .extract_attn import record_activations_attn
from .attach import attach, detach, save, load, train, train_attn
from .calibrate import measure_kl, calibrate_iso_kl
from .method import REGISTRY, register
from . import variants  # noqa: F401  triggers method + config registration

from .variants.mean_diff import MeanDiffConfig, CAAConfig, ActAddConfig, MeanCentredConfig
from .variants.pca import PCAConfig
from .variants.topk_clusters import TopKClustersConfig
from .variants.cosine_gated import CosineGatedConfig
from .variants.sspace import SSpaceConfig
from .variants.spherical import SphericalConfig
from .variants.directional_ablation import DirectionalAblationConfig
from .variants.chars import CHaRSConfig
from .variants.linear_act import LinearAcTConfig
from .variants.angular_steering import AngularSteeringConfig

__all__ = [
    "SteeringConfig",
    "MeanDiffConfig",
    "CAAConfig",
    "ActAddConfig",
    "MeanCentredConfig",
    "PCAConfig",
    "TopKClustersConfig",
    "CosineGatedConfig",
    "SSpaceConfig",
    "SphericalConfig",
    "DirectionalAblationConfig",
    "CHaRSConfig",
    "LinearAcTConfig",
    "AngularSteeringConfig",
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
]
