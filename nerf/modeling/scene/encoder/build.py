import torch

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

from .encoder import Encoder

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Encoder maps low-dimentional input into high-dimentional feature.
"""

def build_encoder(cfg: CfgNode) -> Encoder:
    """
    Build the encoder defined by `cfg.MODEL.ENCODER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    encoder_name = cfg.MODEL.ENCODER.NAME
    encoder = ENCODER_REGISTRY.get(encoder_name)(cfg)
    encoder.to(torch.device(cfg.MODEL.DEVICE))
    log_api_usage("modeling.scene.encoder.{}".format(encoder_name))
    return encoder
