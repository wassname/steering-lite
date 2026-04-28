import os as _os

if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import SteeringConfig
from .extract import record_activations
from .attach import attach, detach, save, load, train
from .calibrate import measure_kl, calibrate_iso_kl
from .method import REGISTRY, register
from . import variants  # noqa: F401  triggers method + config registration

from .variants.mean_diff import MeanDiffConfig
from .variants.pca import PCAConfig
from .variants.topk_clusters import TopKClustersConfig
from .variants.cosine_gated import CosineGatedConfig
from .variants.sspace import SSpaceConfig
from .variants.spherical import SphericalConfig

__all__ = [
    "SteeringConfig",
    "MeanDiffConfig",
    "PCAConfig",
    "TopKClustersConfig",
    "CosineGatedConfig",
    "SSpaceConfig",
    "SphericalConfig",
    "record_activations",
    "train",
    "attach",
    "detach",
    "save",
    "load",
    "measure_kl",
    "calibrate_iso_kl",
    "REGISTRY",
    "register",
]
