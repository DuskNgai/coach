import torch

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

from .sampler import Sampler

SAMPLER_REGISTRY = Registry("SAMPLER")
SAMPLER_REGISTRY.__doc__ = """
Registry for sampler.

Sampler receives a batch of poses and intrinsics, and returns a batch of queries.
"""

def build_sampler(cfg: CfgNode) -> Sampler:
    """
    Build the sampler defined by `cfg.MODEL.SAMPLER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    sampler_name = cfg.MODEL.SAMPLER.NAME
    sampler = SAMPLER_REGISTRY.get(sampler_name)(cfg)
    sampler.to(torch.device(cfg.MODEL.DEVICE))
    log_api_usage("modeling.sampler.{}".format(sampler_name))
    return sampler
