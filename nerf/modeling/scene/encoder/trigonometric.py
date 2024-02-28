import torch

from coach.config import CfgNode

from .build import ENCODER_REGISTRY
from .encoder import Encoder 

class TrigonometricEncoder(Encoder):
    """
    Also called positional encoder. Using a series of `sin` and `cos` functions as encoding functions.
    sin(2 ** 0 * pi * x), ..., sin(2 ** (L - 1) * pi * x), cos(2 ** 0 * pi * x), ..., cos(2 ** (L - 1) * pi * x)
    sin(2 ** 0 * pi * y), ..., sin(2 ** (L - 1) * pi * y), cos(2 ** 0 * pi * y), ..., cos(2 ** (L - 1) * pi * y)
    sin(2 ** 0 * pi * z), ..., sin(2 ** (L - 1) * pi * z), cos(2 ** 0 * pi * z), ..., cos(2 ** (L - 1) * pi * z)
    """

    def __init__(self,
        n_frequencies: int = 10
    ) -> None:
        super().__init__()

        self.n_frequencies = n_frequencies
        assert self.n_frequencies > 0, "n_frequencies should be a positive integer."

        # frequencies.size() = [self.n_frequencies]
        frequencies = torch.logspace(0.0, self.n_frequencies - 1, self.n_frequencies, 2.0)
        self.frequencies = frequencies * torch.pi

    def forward(self, tensor_in: torch.Tensor) -> torch.Tensor:
        """
        See also: `Encoder.forward()`.
        """
        # Out product of tensor_in and self.frequencies.
        tensor_out = torch.einsum("...i,j->...ij", tensor_in, self.frequencies).flatten(-2)
        tensor_out = torch.cat([tensor_out, torch.sin(tensor_out), torch.cos(tensor_out)], dim=-1)
        return tensor_out

@ENCODER_REGISTRY.register()
def build_trigonometric_encoder(cfg: CfgNode) -> TrigonometricEncoder:
    """
    Build the trigonometric encoder defined by `cfg`.
    """
    encoder = TrigonometricEncoder(
        n_frequencies=cfg.MODEL.ENCODER.TRIGONOMETRIC.N_FREQUENCIES,
    )
    return encoder
