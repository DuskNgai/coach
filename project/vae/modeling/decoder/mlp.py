from typing import List, Optional, Union

import torch.nn as nn

from coach.config import CfgNode
from coach.modeling.layer import Mlp

from .build import VAE_DECODER_REGISTRY

class MlpDecoder(Mlp):
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
        self.layers.append(nn.Unflatten(1, (1, image_size, image_size)))


@VAE_DECODER_REGISTRY.register()
def build_ae_mlp_decoder(cfg: CfgNode) -> MlpDecoder:
    """
    Build a multi-layer perceptron (MLP) decoder.
    """
    return MlpDecoder(
        image_size=cfg.MODEL.IMAGE_SIZE,
        in_channels=cfg.MODEL.LATENT_CHANNELS,
        hidden_layers=cfg.MODEL.DECODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.DECODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.IN_CHANNELS,
        bias=cfg.MODEL.DECODER.BIAS,
        act_layer=nn.ReLU,
    )

@VAE_DECODER_REGISTRY.register()
def build_vae_mlp_decoder(cfg: CfgNode) -> MlpDecoder:
    """
    Build a multi-layer perceptron (MLP) decoder.
    """
    return MlpDecoder(
        image_size=cfg.MODEL.IMAGE_SIZE,
        in_channels=cfg.MODEL.LATENT_CHANNELS,
        hidden_layers=cfg.MODEL.DECODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.DECODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.IN_CHANNELS,
        bias=cfg.MODEL.DECODER.BIAS,
        act_layer=nn.ReLU,
    )
