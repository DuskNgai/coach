import torch

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

from .encoder import Encoder

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Encoder maps low-dimentional input into high-dimentional feature.
"""

def build_encoder(cfg: CfgNode, encoder_name: str, device: str) -> Encoder:
    """
    Build the encoder defined by `cfg`.
    It does not load checkpoints from `cfg`.
    """
    encoder_name = cfg.NAME
    encoder = ENCODER_REGISTRY.get(encoder_name)(cfg)
    encoder.to(torch.device(device))
    log_api_usage("nerf.modeling.scene.encoder.{}".format(encoder_name))
    return encoder

def build_positional_encoder(cfg: CfgNode) -> Encoder:
    """
    Build the encoder defined by `cfg.MODEL.SCENE.POSITIONAL_ENCODER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    return build_encoder(
        cfg.MODEL.SCENE.POSITIONAL_ENCODER,
        cfg.MODEL.SCENE.POSITIONAL_ENCODER.NAME,
        cfg.MODEL.DEVICE
    )

def build_directional_encoder(cfg: CfgNode) -> Encoder:
    """
    Build the encoder defined by `cfg.MODEL.SCENE.DIRECTIONAL_ENCODER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    return build_encoder(
        cfg.MODEL.SCENE.DIRECTIONAL_ENCODER,
        cfg.MODEL.SCENE.DIRECTIONAL_ENCODER.NAME,
        cfg.MODEL.DEVICE
    )
