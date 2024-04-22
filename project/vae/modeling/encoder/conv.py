from typing import List, Optional, Union

import torch.nn as nn

from coach.config import CfgNode
from coach.modeling.layer import ConvNet

from .build import VAE_ENCODER_REGISTRY

class ConvEncoder(ConvNet):
    def __init__(self,
        image_size: int,
        in_channels: int,
        hidden_layers: int,
        hidden_channels: Optional[Union[int, List[int]]],
        out_channels: int,
        bias: bool,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        act_layer: nn.Module = nn.ReLU
    ) -> None:
        super().__init__(
            in_channels,
            hidden_layers - 1,
            hidden_channels[:-1],
            hidden_channels[-1],
            bias=bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_layer=act_layer
        )
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear((image_size // 2 ** (hidden_layers - 2)) ** 2 * hidden_channels[-1], out_channels, bias=bias))


@VAE_ENCODER_REGISTRY.register()
def build_ae_conv_encoder(cfg: CfgNode) -> ConvEncoder:
    """
    Build a multi-layer convolutional encoder.
    """
    return ConvEncoder(
        image_size=cfg.MODEL.IMAGE_SIZE,
        in_channels=cfg.MODEL.IN_CHANNELS,
        hidden_layers=cfg.MODEL.ENCODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.ENCODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.LATENT_CHANNELS,
        bias=cfg.MODEL.ENCODER.BIAS,
        act_layer=nn.ReLU,
    )

@VAE_ENCODER_REGISTRY.register()
def build_vae_conv_encoder(cfg: CfgNode) -> ConvEncoder:
    """
    Build a multi-layer convolutional encoder.
    """
    return ConvEncoder(
        image_size=cfg.MODEL.IMAGE_SIZE,
        in_channels=cfg.MODEL.IN_CHANNELS,
        hidden_layers=cfg.MODEL.ENCODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.ENCODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.LATENT_CHANNELS * 2,
        bias=cfg.MODEL.ENCODER.BIAS,
        act_layer=nn.ReLU,
    )
