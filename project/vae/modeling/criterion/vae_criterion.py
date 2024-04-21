import torch
import torch.nn as nn

from coach.modeling.criterion import Criterion, CRITERION_REGISTRY

class VAECriterion(Criterion):
    """
    The criterion for the variational autoencoder.
    The loss is the sum of the reconstruction loss and the KL divergence.
    """

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight
        self.loss = nn.MSELoss(reduction="none")

    def forward(self,
            images: torch.Tensor,
            recons: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): The ground truth images.
            recons (torch.Tensor): The reconstructed images.
            mu (torch.Tensor): The mean of the latent representation.
            logvar (torch.Tensor): The log variance of the latent representation.
        """
        reconstruction_loss = self.loss(recons, images).sum(-1).mean()
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return {
            "loss": self.weight * reconstruction_loss + kl_divergence,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence_loss": kl_divergence
        }


@CRITERION_REGISTRY.register()
def build_vae_criterion(cfg: dict) -> VAECriterion:
    """
    Build the VAE criterion.
    """
    return VAECriterion(weight=cfg.MODEL.CRITERION.WEIGHT)