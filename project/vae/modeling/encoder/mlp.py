from typing import List, Optional, Union

import torch
import torch.nn as nn

from coach.config import CfgNode
from coach.modeling.layer import Mlp

from .build import VAE_ENCODER_REGISTRY

class MlpEncoder(Mlp):
    def __init__(self,
        in_channels: int,
        hidden_layers: int,
        hidden_channels: Optional[Union[int, List[int]]],
        out_channels: int,
        bias: bool,
        act_layer: nn.Module = nn.ReLU
    ) -> None:
        super().__init__(in_channels, hidden_layers, hidden_channels, out_channels, bias, act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


@VAE_ENCODER_REGISTRY.register()
def build_ae_mlp_encoder(cfg: CfgNode) -> MlpEncoder:
    """
    Build a multi-layer perceptron (MLP) encoder.
    """
    return MlpEncoder(
        in_channels=cfg.MODEL.IN_CHANNELS,
        hidden_layers=cfg.MODEL.ENCODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.ENCODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.LATENT_CHANNELS,
        bias=cfg.MODEL.ENCODER.BIAS,
        act_layer=nn.ReLU,
    )

@VAE_ENCODER_REGISTRY.register()
def build_vae_mlp_encoder(cfg: CfgNode) -> MlpEncoder:
    """
    Build a multi-layer perceptron (MLP) encoder.
    """
    return MlpEncoder(
        in_channels=cfg.MODEL.IN_CHANNELS,
        hidden_layers=cfg.MODEL.ENCODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.ENCODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.LATENT_CHANNELS * 2,
        bias=cfg.MODEL.ENCODER.BIAS,
        act_layer=nn.ReLU,
    )
