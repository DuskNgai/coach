import torch

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from fvcore.common.registry import Registry


VAE_ENCODER_REGISTRY = Registry("VAE_ENCODER")
VAE_ENCODER_REGISTRY.__doc__ = """
The encoder in VAE that may output mean and variance.
"""

def build_vae_encoder(cfg: CfgNode) -> torch.nn.Module:
    """
    Build the encoder defined by `cfg.MODEL.ENCODER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    vae_encoder_name = cfg.MODEL.ENCODER.NAME
    vae_encoder = VAE_ENCODER_REGISTRY.get(vae_encoder_name)(cfg)
    log_api_usage("vae.modeling.encoder.{}".format(vae_encoder_name))
    return vae_encoder
