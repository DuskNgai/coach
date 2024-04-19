from typing import Any

import torch.nn as nn

from coach.config import configurable, CfgNode
from coach.modeling.architecture import MODEL_REGISTRY
from coach.modeling.criterion import build_criterion, Criterion

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
        encoder = build_encoder(cfg)
        decoder = build_decoder(cfg)
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
        latent = self.encoder(batched_inputs)
        outputs = self.decoder(latent)
        losses = self.criterion(outputs, inputs)
        return losses
