import torch

from coach.config import CfgNode

from .build import NERF_RENDERER_REGISTRY
from .renderer import Renderer

__all__ = [
    "VolumeRenderer",
    "build_volume_renderer"
]


class VolumeRenderer(Renderer):
    """
    A regular sampling volume renderer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, attributes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            attributes (dict[str, torch.Tensor]):
                A dict containing the following keys:
                - xyz (torch.Tensor): The coordinates of the points. [B, T, 3]
                - dir (torch.Tensor): The directions of the rays. [B, T, 3]
                - density (torch.Tensor): The densities of the points. [B, T]
                - color (torch.Tensor): The colors of the points. [B, T, 3]
        """
        B, T, _ = attributes["xyz"].shape

        xyzs = attributes["xyz"] # [B, T, 3]
        densities = attributes["density"] # [B, T]
        colors = attributes["color"] # [B, T, 3]

        steps = torch.cat([xyzs.diff(dim=-2).norm(dim=-1), xyzs.new_zeros((B, 1))]) # [B, T]
        alphas = 1.0 - torch.exp(-densities * steps) # [B, T]
        transmittances = torch.cumprod(1.0 - alphas, dim=-1) # [B, T]
        weights = alphas * transmittances # [B, T]
        rgbs = torch.sum(weights[..., None] * colors, dim=-2) # [B, 3]

        outputs = { "rgb": rgbs }

        if self.require_depth:
            depths = torch.sum(weights * steps, dim=-1, keepdim=True) # [B, 1]
            outputs["depth"] = depths

        return outputs

@NERF_RENDERER_REGISTRY.register()
def build_volume_renderer(cfg: CfgNode) -> VolumeRenderer:
    """
    Build the volume renderer defined by `cfg.MODEL.RENDERER.NAME`.
    """
    renderer = VolumeRenderer()
    return renderer
