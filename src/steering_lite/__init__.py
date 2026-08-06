import os as _os

if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import SteeringConfig, REGISTRY, register
from .extract import record_activations
from .attach import attach, detach, save, load, train
from .calibrate import measure_kl, calibrate_iso_kl
from .word_readout import (readout_words, format_readout, readout_effect, format_effect,
                           midpoint_states, readout_at_point, format_at_point)
from .tuned_lens import IdentityLens, JacobianLens, TunedLens
from . import variants  # noqa: F401  triggers method + config registration
from .vector import Vector

from .variants.mean_diff import MeanDiffC
from .variants.pca import PCAC
from .variants.topk_clusters import TopKClustersC
from .variants.cosine_gated import CosineGatedC
from .variants.sspace import SSpaceC
from .variants.sspace_pca import SSpacePCAC
from .variants.corda_pca import CordaPCAC
from .variants.sspace_ablate import SSpaceAblateC
from .variants.sspace_damp_amp import SSpaceDampAmpC
from .variants.super_sspace import SuperSSpaceC
from .variants.spherical import SphericalC
from .variants.directional_ablation import DirectionalAblationC
from .variants.chars import CHaRSC
from .variants.linear_act import LinearAcTC
from .variants.angular_steering import AngularSteeringC
from .variants.random import RandomC
from .variants.vjp_delta import VjpDeltaC

__all__ = [
    "SteeringConfig",
    "MeanDiffC",
    "PCAC",
    "TopKClustersC",
    "CosineGatedC",
    "SSpaceC",
    "SSpacePCAC",
    "CordaPCAC",
    "SSpaceAblateC",
    "SSpaceDampAmpC",
    "SuperSSpaceC",
    "SphericalC",
    "DirectionalAblationC",
    "CHaRSC",
    "LinearAcTC",
    "AngularSteeringC",
    "RandomC",
    "VjpDeltaC",
    "record_activations",
    "train",
    "attach",
    "detach",
    "save",
    "load",
    "measure_kl",
    "calibrate_iso_kl",
    "readout_words",
    "format_readout",
    "readout_effect",
    "format_effect",
    "midpoint_states",
    "readout_at_point",
    "format_at_point",
    "TunedLens",
    "JacobianLens",
    "IdentityLens",
    "REGISTRY",
    "register",
    "Vector",
]
