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
from .variants.kv_cache_gram import KVCacheGramC
from .variants.vjp_delta import VjpDeltaC
from .variants.vjp_cache import VjpCacheC
from .variants.query_steer import QuerySteerC
from .variants.attn_site import KeySteerC, ValueSteerC, QVjpC, KVjpC, QRetrieveC, QRSumC, QRetrieveDeltaC, QRetrSumC, SinkWriteC, SinkValueC, SinkRSumC, SinkPunctC, QPrefixC, SinkRRandC, QPrefixKC, QPrefixK0C

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
    "KVCacheGramC",
    "VjpDeltaC",
    "VjpCacheC",
    "QuerySteerC",
    "KeySteerC",
    "ValueSteerC",
    "QVjpC",
    "KVjpC",
    "QRetrieveC",
    "QRSumC",
    "QRetrieveDeltaC",
    "QRetrSumC",
    "SinkWriteC",
    "SinkValueC",
    "SinkRSumC",
    "SinkPunctC",
    "QPrefixC",
    "SinkRRandC",
    "QPrefixKC",
    "QPrefixK0C",
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
