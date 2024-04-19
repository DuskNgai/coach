from typing import Any

import torch.nn as nn

from coach.config import configurable, CfgNode
from coach.modeling.architecture import MODEL_REGISTRY
from coach.modeling.criterion import build_criterion, Criterion
from project.nerf.modeling.sampler import build_sampler, Sampler
from project.nerf.modeling.scene import build_scene, Scene
from project.nerf.modeling.renderer import build_renderer, Renderer

__all__ = ["NeRF"]


@MODEL_REGISTRY.register()
class NeRF(nn.Module):
    """
    The NeRF. A model that contains the following components:
    1. Sampler: Samples rays from the input images.
    2. Scene: Computes the attributes of the scene.
    3. Renderer: Renders the attributes into the output images.
    4. Criterion: Computes the loss between the rendered images and the ground truth images.
    """

    @configurable
    def __init__(
        self,
        sampler: Sampler,
        scene: Scene,
        renderer: Renderer,
        criterion: Criterion,
    ) -> None:
        super().__init__()

        self.sampler = sampler
        self.scene = scene
        self.renderer = renderer
        self.criterion = criterion

    @classmethod
    def from_config(cls, cfg: CfgNode) -> dict[str, Any]:
        sampler = build_sampler(cfg)
        scene = build_scene(cfg)
        renderer = build_renderer(cfg)
        criterion = build_criterion(cfg)
        return {
            "sampler": sampler,
            "scene": scene,
            "renderer": renderer,
            "criterion": criterion
        }

    def forward(self, batched_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            batched_inputs (dict[str, Any]): A batch of inputs.
        """
        queries = self.sampler(batched_inputs)
        attributes = self.scene(queries)
        outputs = self.renderer(attributes)
        losses = self.criterion(outputs)
        return losses

    def inference(self, batched_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            batched_inputs (dict[str, Any]): A batch of inputs.
        """
        queries = self.sampler(batched_inputs)
        attributes = self.scene(queries)
        outputs = self.renderer(attributes)
        return outputs
