from typing import Any

import torch

from coach.config import CfgNode

from .build import NERF_SAMPLER_REGISTRY
from .sampler import Sampler, Projection

class NearFarSampler(Sampler):
    """
    """

    def __init__(self,
        near: float,
        far: float,
        projection: Projection
    ) -> None:
        super().__init__()

        self.near = near
        self.far = far
        self.projection = projection

    def forward(self, batched_inputs: dict[str, Any]) -> None:
        if self.projection == Projection.Perspective:
            pass
        elif self.projection == Projection.Orthographic:
            pass
        else:
            raise ValueError("Invalid projection type: {}".format(self.projection))

    def sample_from_near_far(self):
        ts = torch.rand(num_samples, num_steps)
        samples = (self.far - self.near) * ts + self.near
        return samples

    def perspective_sample(self, poses: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:

        queries = {
            "xyz": torch.rand(1, 3),
            "dir": torch.rand(1, 3)
        }

        return queries
    
    def orthographic_sample(self, poses: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        uv = torch.meshgrid(
            torch.linspace(-1, 1, 100),
            torch.linspace(-1, 1, 100),
            indexing="ij"
        )

        queries = {
            "xyz": torch.rand(1, 3),
            "dir": torch.rand(1, 3)
        }

        return queries


@NERF_SAMPLER_REGISTRY.register()
def build_near_far_sampler(cfg: CfgNode) -> NearFarSampler:
    """
    Build the near far sampler defined by `cfg.MODEL.SAMPLER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    sampler = NearFarSampler(
        Projection(cfg.MODEL.SAMPLER.PROJECTION.lower())
    )
    return sampler
