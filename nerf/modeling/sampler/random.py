import torch

from coach.config import CfgNode

from .build import SAMPLER_REGISTRY
from .sampler import Sampler

class RandomSampler(Sampler):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> None:
        pass

@SAMPLER_REGISTRY.register()
def build_random_sampler(cfg: CfgNode) -> RandomSampler:
    """
    Build the random sampler defined by `cfg.MODEL.SAMPLER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    sampler = RandomSampler()
    return sampler
