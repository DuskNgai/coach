from typing import Any, Dict

import torch
import torch.nn as nn

from coach.config import configurable, CfgNode
from coach.modeling.architecture import MODEL_REGISTRY
from coach.modeling.criterion import build_criterion, Criterion
from project.vae.modeling.encoder import build_vae_encoder
from project.vae.modeling.decoder import build_vae_decoder

from .ae import AutoEncoder

__all__ = ["VariationalAutoEncoder"]


@MODEL_REGISTRY.register()
class VariationalAutoEncoder(AutoEncoder):
    """
    A vanilla variational autoencoder. A model that contains the following components:
    1. Encoder: Encodes the input into a latent representation, which is sample from its mean and variance.
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
        super().__init__(
            device=device,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion
        )

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
        images = images.flatten(1)
        encoded = self.encoder(images)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        latent = self.reparameterize(mu, logvar)
        outputs = self.decoder(latent)
        outputs = torch.sigmoid(outputs)
        losses = self.criterion(images, outputs, mu, logvar)
        return losses

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize the latent representation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def inference(self, latent: torch.Tensor) -> torch.Tensor:
        outputs = self.decoder(latent)
        outputs = torch.sigmoid(outputs)
        return outputs
