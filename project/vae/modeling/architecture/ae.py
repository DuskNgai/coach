from typing import Any, Dict

import torch
import torch.nn as nn

from coach.config import configurable, CfgNode
from coach.modeling.architecture import MODEL_REGISTRY
from coach.modeling.criterion import build_criterion, Criterion
from project.vae.modeling.encoder import build_vae_encoder
from project.vae.modeling.decoder import build_vae_decoder

__all__ = ["AutoEncoder"]


@MODEL_REGISTRY.register()
class AutoEncoder(nn.Module):
    """
    A vanilla autoencoder. A model that contains the following components:
    1. Encoder: Encodes the input into a latent representation.
    2. Decoder: Decodes the latent representation into the output.
    3. Criterion: Computes the loss between the output and the input.
    """

    @configurable
    def __init__(
        self,
        device: torch.device,
        encoder: nn.Module,
        decoder: nn.Module,
        criterion: Criterion,
    ) -> None:
        super().__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

    @classmethod
    def from_config(cls, cfg: CfgNode) -> dict[str, nn.Module]:
        encoder = build_vae_encoder(cfg)
        decoder = build_vae_decoder(cfg)
        criterion = build_criterion(cfg)
        return {
            "device": torch.device(cfg.MODEL.DEVICE),
            "encoder": encoder,
            "decoder": decoder,
            "criterion": criterion
        }

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            images (torch.Tensor): The input images.
        """
        images = images.to(self.device)
        latent = self.encoder(images.flatten(1))
        outputs = self.decoder(latent)
        losses = self.criterion(outputs.reshape_as(images), images)
        return {"loss": losses}
