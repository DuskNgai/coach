import torch
import torch.nn as nn

from coach.config import CfgNode

from .build import SCENE_REGISTRY
from .scene import Scene
from .encoder import build_encoder

__all__ = ["VanillaScene"]

class VanillaScene(Scene):

    def __init__(self,
        positional_encoder: nn.Module,
        directional_encoder: nn.Module,
        mlp_width: int = 256,
        mlp_depth: int = 8,
        skip_connection: bool = True
    ):
        super().__init__()

        self.positional_encoder = positional_encoder
        self.directional_encoder = directional_encoder

        # #* Stage 1 of NeRF MLP.
        # self.stage1 = build_mlp(
        #     config["network"]["stage_1_network"],
        #     in_dim=self.pos_encoder.out_dim
        # )

        # #* Stage 2 of NeRF MLP, starting from a skip connection.
        # self.stage2 = build_mlp(
        #     config["network"]["stage_2_network"],
        #     in_dim=self.pos_encoder.out_dim + config["network"]["stage_2_network"]["mlp_width"]
        # )

        # #* Network for predicting density.
        # self.density_net = build_mlp(
        #     config["network"]["density_network"],
        #     out_dim=1
        # )

        # #* Network for connecting stage2 and rgb.
        # self.feature_net = build_mlp(
        #     config["network"]["feature_network"]
        # )

        # #* Network for predicting rgb, no activation function after the last layer.
        # self.rgb_net = build_mlp(
        #     config["network"]["rgb_network"],
        #     in_dim=self.dir_encoder.out_dim + config["network"]["feature_network"]["mlp_width"],
        #     out_dim=3
        # )

    def forward(self, queries: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            queries (dict[str, torch.Tensor]):
                A dict containing the following keys:
                - xyz (torch.Tensor): The coordinates of the points. [B, T, 3]
                - dir (torch.Tensor): The directions of the rays. [B, T, 3]
        """

        xyzs = queries["xyz"]
        dirs = queries["dir"]

        encoded_pos = self.pos_encoder(xyzs)
        encoded_dir = self.dir_encoder(dirs)

        feature_1 = self.stage1(encoded_pos)
        feature_2 = self.stage2(torch.cat([encoded_pos, feature_1], dim=-1))

        densities = self.density_net(feature_2)

        feature_before_rgb = self.feature_net(feature_2)
        colors = self.rgb_net(torch.cat([encoded_dir, feature_before_rgb], dim=-1))

        attributes = {
            "xyz": xyzs,
            "dir": dirs,
            "density": densities,
            "color": colors
        }
        return attributes

@SCENE_REGISTRY.register()
def build_vanilla_scene(cfg: CfgNode) -> VanillaScene:
    """Build the vanilla scene defined by `cfg.MODEL.SCENE.NAME`.
    It does not load checkpoints from `cfg`.
    """
    scene = VanillaScene(
        positional_encoder=build_encoder(cfg.MODEL.SCENE.POSITIONAL_ENCODER),
        directional_encoder=build_encoder(cfg.MODEL.SCENE.DIRECTIONAL_ENCODER),
        mlp_width=cfg.MODEL.SCENE.MLP_WIDTH,
        mlp_depth=cfg.MODEL.SCENE.MLP_DEPTH,
        skip_connection=cfg.MODEL.SCENE.SKIP_CONNECTION
    )
    return scene
