from typing import List, Optional, Union

import torch.nn as nn

from coach.config import CfgNode
from coach.modeling.layer import ConvTransposeNet

from .build import VAE_DECODER_REGISTRY

class ConvDecoder(ConvTransposeNet):
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
            hidden_channels[0],
            hidden_layers - 1,
            hidden_channels[1:],
            out_channels,
            bias=bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_layer=act_layer
        )
        self.layers.insert(0, nn.Linear(in_channels, (image_size // 2 ** (hidden_layers - 2)) ** 2 * hidden_channels[0], bias=bias))
        self.layers.insert(1, nn.ReLU())
        self.layers.insert(2, nn.Unflatten(1, (hidden_channels[0], image_size // 2 ** (hidden_layers - 2), image_size // 2 ** (hidden_layers - 2))))


@VAE_DECODER_REGISTRY.register()
def build_ae_conv_decoder(cfg: CfgNode) -> ConvDecoder:
    """
    Build a multi-layer convolutional decoder.
    """
    return ConvDecoder(
        image_size=cfg.MODEL.IMAGE_SIZE,
        in_channels=cfg.MODEL.LATENT_CHANNELS,
        hidden_layers=cfg.MODEL.DECODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.DECODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.IN_CHANNELS,
        bias=cfg.MODEL.DECODER.BIAS,
        act_layer=nn.ReLU,
    )

@VAE_DECODER_REGISTRY.register()
def build_vae_conv_decoder(cfg: CfgNode) -> ConvDecoder:
    """
    Build a multi-layer convolutional decoder.
    """
    return ConvDecoder(
        image_size=cfg.MODEL.IMAGE_SIZE,
        in_channels=cfg.MODEL.LATENT_CHANNELS,
        hidden_layers=cfg.MODEL.DECODER.HIDDEN_LAYERS,
        hidden_channels=cfg.MODEL.DECODER.HIDDEN_CHANNELS,
        out_channels=cfg.MODEL.IN_CHANNELS,
        bias=cfg.MODEL.DECODER.BIAS,
        act_layer=nn.ReLU,
    )
