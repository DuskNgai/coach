from typing import Any

import torch

from coach.config import CfgNode

from .build import NERF_SAMPLER_REGISTRY
from .sampler import Sampler, Projection

class BBoxSampler(Sampler):
    """
    """

    def __init__(self, projection: Projection) -> None:
        super().__init__()

        self.projection = projection

    def forward(self, batched_inputs: dict[str, Any]) -> None:
        if self.projection == Projection.Perspective:
            pass
        elif self.projection == Projection.Orthographic:
            pass
        else:
            raise ValueError("Invalid projection type: {}".format(self.projection))

    def perspective_sample(self, poses: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        queries = {
            "xyz": torch.rand(1, 3),
            "dir": torch.rand(1, 3)
        }

        return queries

    def orthographic_sample(self, poses: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        queries = {
            "xyz": torch.rand(1, 3),
            "dir": torch.rand(1, 3)
        }

        return queries


@NERF_SAMPLER_REGISTRY.register()
def build_bbox_sampler(cfg: CfgNode) -> BBoxSampler:
    """
    Build the bounding box sampler defined by `cfg.MODEL.SAMPLER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    sampler = BBoxSampler(
        Projection(cfg.MODEL.SAMPLER.PROJECTION.lower())
    )
    return sampler
