import torch

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

from .renderer import Renderer

NERF_RENDERER_REGISTRY = Registry("NERF_RENDERER")
NERF_RENDERER_REGISTRY.__doc__ = """
Composite the attributes to obtain the final rendering result (color, depth ...).
"""

def build_renderer(cfg: CfgNode) -> Renderer:
    """
    Build the renderer defined by `cfg.MODEL.RENDERER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    renderer_name = cfg.MODEL.RENDERER.NAME
    renderer = NERF_RENDERER_REGISTRY.get(renderer_name)(cfg)
    renderer.to(torch.device(cfg.MODEL.DEVICE))
    log_api_usage("nerf.modeling.renderer.{}".format(renderer_name))
    return renderer
