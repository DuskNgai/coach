import torch
import torch.nn as nn
import torch.nn.functional as F

from coach.config import CfgNode

from .build import NERF_ENCODER_REGISTRY
from .encoder import Encoder

class GridEncoder(Encoder):

    def __init__(self,
        resolution: int,
        out_channels: int = 1
    ):
        super().__init__()
        self.out_channels = out_channels
        self.grid = nn.Parameter(
            torch.zeros(out_channels, resolution, resolution, resolution),
            requires_grad=True
        )

    @property
    def out_channels(self) -> int:
        """
        See also: `Encoder.out_channels`.
        """
        return self.out_channels

    def forward(self, tensor_in: torch.Tensor) -> torch.Tensor:
        tensor_out = F.grid_sample(
            tensor_in,
            self.grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        return tensor_out

@NERF_ENCODER_REGISTRY.register()
def build_grid_encoder(cfg: CfgNode) -> GridEncoder:
    """
    Build the grid encoder defined by `cfg`.
    """
    encoder = GridEncoder(
        resolution=cfg.RESOLUTION,
        out_channels=cfg.OUT_CHANNELS
    )
    return encoder
