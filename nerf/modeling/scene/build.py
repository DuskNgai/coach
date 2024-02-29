import torch

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

from .scene import Scene

SCENE_REGISTRY = Registry("SCENE")
SCENE_REGISTRY.__doc__ = """
Registry for scene, i.e. the 3D world.
It can be neural networks or grid-based representations.

One can sample point locations from a scene,
and obtain the corresponding attributes (e.g. color, normal, etc.).
"""

def build_scene(cfg: CfgNode) -> Scene:
    """
    Build the scene defined by `cfg.MODEL.SCENE.NAME`.
    It does not load checkpoints from `cfg`.
    """
    scene_name = cfg.MODEL.SCENE.NAME
    scene = SCENE_REGISTRY.get(scene_name)(cfg)
    scene.to(torch.device(cfg.MODEL.DEVICE))
    log_api_usage("nerf.modeling.scene.{}".format(scene_name))
    return scene
