from typing import Any

import torch
import torch.nn as nn

from coach.config import configurable, CfgNode
from coach.modeling.architecture import MODEL_REGISTRY
from coach.modeling.criterion import build_criterion, Criterion
from project.vae.modeling.encoder import build_vae_encoder
from project.vae.modeling.decoder import build_vae_decoder

__all__ = ["VariationalAutoEncoder"]


@MODEL_REGISTRY.register()
class VariationalAutoEncoder(nn.Module):
    """
    A vanilla variational autoencoder. A model that contains the following components:
    1. Encoder: Encodes the input into a latent representation, which is sample from its mean and variance.
    2. Decoder: Decodes the latent representation into the output.
    3. Criterion: Computes the loss between the output and the input.
    """

    @configurable
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        criterion: Criterion,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

    @classmethod
    def from_config(cls, cfg: CfgNode) -> dict[str, nn.Module]:
        encoder = build_vae_encoder(cfg)
        decoder = build_vae_decoder(cfg)
        criterion = build_criterion(cfg)
        return {
            "encoder": encoder,
            "decoder": decoder,
            "criterion": criterion
        }
    
    def forward(self, batched_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            batched_inputs (dict[str, Any]): A batch of inputs.
        """
        mu, logvar = self.encoder(batched_inputs)
        latent = self.reparameterize(mu, logvar)
        outputs = self.decoder(latent)
        losses = self.criterion(outputs, inputs)
        return losses

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize the latent representation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
