import torch
import torch.nn as nn

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

VAE_DECODER_REGISTRY = Registry("VAE_DECODER")
VAE_DECODER_REGISTRY.__doc__ = """
The decoder in VAE that output the reconstructed input.
"""

def build_vae_decoder(cfg: CfgNode) -> nn.Module:
    """
    Build the decoder defined by `cfg.MODEL.DECODER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    vae_decoder_name = cfg.MODEL.DECODER.NAME
    vae_decoder = VAE_DECODER_REGISTRY.get(vae_decoder_name)(cfg)
    log_api_usage("vae.modeling.decoder.{}".format(vae_decoder_name))
    return vae_decoder
