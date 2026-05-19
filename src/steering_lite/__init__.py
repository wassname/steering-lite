import os as _os

if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import SteeringConfig, REGISTRY, register
from .extract import record_activations
from .attach import attach, detach, save, load, train
from .calibrate import measure_kl, calibrate_iso_kl
from . import variants  # noqa: F401  triggers method + config registration
from .vector import Vector

from .variants.mean_diff import MeanDiffC
from .variants.pca import PCAC
from .variants.topk_clusters import TopKClustersC
from .variants.cosine_gated import CosineGatedC
from .variants.sspace import SSpaceC
from .variants.sspace_ablate import SSpaceAblateC
from .variants.sspace_damp_amp import SSpaceDampAmpC
from .variants.super_sspace import SuperSSpaceC
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
    "SSpaceAblateC",
    "SSpaceDampAmpC",
    "SuperSSpaceC",
    "SphericalC",
    "DirectionalAblationC",
    "CHaRSC",
    "LinearAcTC",
    "AngularSteeringC",
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
    "Vector",
]
