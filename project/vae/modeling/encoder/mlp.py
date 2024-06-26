from typing import List, Optional, Union

import torch.nn as nn

from coach.config import CfgNode
from coach.modeling.layer import Mlp

from .build import VAE_ENCODER_REGISTRY

class MlpEncoder(Mlp):
    def __init__(self,
        image_size: int,
        in_channels: int,
        hidden_layers: int,
        hidden_channels: Optional[Union[int, List[int]]],
        out_channels: int,
        bias: bool,
        act_layer: nn.Module = nn.ReLU
    ) -> None:
        super().__init__(in_channels, hidden_layers, hidden_channels, out_channels, bias, act_layer)
        self.layers.insert(0, nn.Flatten())


@VAE_ENCODER_REGISTRY.register()
def build_ae_mlp_encoder(cfg: CfgNode) -> MlpEncoder:
    """
    Build a multi-layer perceptron (MLP) encoder.
    """
    return MlpEncoder(
        image_size=cfg.MODEL.IMAGE_SIZE,
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
        image_size=cfg.MODEL.IMAGE_SIZE,
        in_channels=cfg.MODEL.IN_CHANNELS,
        hidden_layers=cfg.MODEL.ENCODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.ENCODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.LATENT_CHANNELS * 2,
        bias=cfg.MODEL.ENCODER.BIAS,
        act_layer=nn.ReLU,
    )
