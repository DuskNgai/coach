import torch

from .encoder import Encoder
from .sh_utils import generate_sh_funcs

class SphericalHarmonicEncoder(Encoder):
    """
    Using a set of spherical harmonics as encoding functions. 
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.n_degrees = config["n_degrees"]

        self.enc_funcs = generate_sh_funcs(self.n_degrees)

    def forward(self, tensor_in: torch.Tensor) -> torch.Tensor:
        x, y, z = torch.split(tensor_in, 1, dim=-1)
        tensor_out = torch.cat(self.enc_funcs(x, y, z), dim=-1)
        return tensor_out
