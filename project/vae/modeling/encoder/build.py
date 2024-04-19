import torch
import torch.nn as nn

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

VAE_ENCODER = Registry("VAE_ENCODER")
VAE_ENCODER.__doc__ = """
The encoder in VAE that may output mean and variance.
"""

def build_vae_encoder(cfg: CfgNode) -> nn.Module:
    """
    Build the encoder defined by `cfg.MODEL.ENCODER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    vae_encoder_name = cfg.MODEL.ENCODER.NAME
    vae_encoder = VAE_ENCODER.get(vae_encoder_name)(cfg)
    vae_encoder.to(torch.device(cfg.MODEL.DEVICE))
    log_api_usage("vae.modeling.encoder.{}".format(vae_encoder_name))
    return vae_encoder
