import torch
import torch.nn as nn

from coach.config import CfgNode
from coach.modeling.layer import Mlp

from .build import NERF_SCENE_REGISTRY
from .scene import Scene
from .encoder import build_positional_encoder, build_directional_encoder, Encoder

__all__ = [
    "VanillaScene",
    "build_vanilla_scene",
]


class VanillaScene(Scene):

    def __init__(self,
        positional_encoder: Encoder,
        directional_encoder: Encoder,
        mlp_width: int = 256,
        mlp_depth: int = 8,
        skip_connection: bool = True
    ) -> None:
        super().__init__()
        act_layer = nn.ReLU
        self.skip_connection = skip_connection

        self.positional_encoder = positional_encoder
        self.directional_encoder = directional_encoder
        self.act_layer = act_layer()

        if skip_connection:
            #* Stage 1 of NeRF MLP.
            self.stage1 = Mlp(
                in_channels=self.positional_encoder.out_channels,
                hidden_layers=5 - 1,
                hidden_channels=mlp_width,
                act_layer=act_layer,
            )

            #* Stage 2 of NeRF MLP, starting from a skip connection.
            self.stage2 = Mlp(
                in_channels=self.positional_encoder.out_channels + mlp_width,
                hidden_layers=mlp_depth - 2 - (5 - 1),
                hidden_channels=mlp_width,
                act_layer=act_layer,
            )
        else:
            self.stage = Mlp(
                in_channels=self.positional_encoder.out_channels,
                hidden_layers=mlp_depth - 1,
                hidden_channels=mlp_width,
                act_layer=act_layer,
            )

        #* Network for predicting density.
        self.density_net = Mlp(
            in_channels=mlp_width,
            hidden_layers=0,
            out_channels=1,
            act_layer=act_layer,
        )

        #* Network for connecting stage2 and rgb.
        self.feature_net = Mlp(
            in_channels=mlp_width,
            hidden_layers=0,
            out_channels=mlp_width // 2,
            act_layer=act_layer,
        )

        #* Network for predicting rgb, no activation function after the last layer.
        self.rgb_net = Mlp(
            in_channels=self.directional_encoder.out_channels + mlp_width // 2,
            hidden_layers=1,
            hidden_channels=mlp_width // 2,
            out_channels=3,
            act_layer=act_layer
        )

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

        if self.skip_connection:
            network_feature = self.act_layer(self.stage1(encoded_pos))
            feature = self.act_layer(self.stage2(torch.cat([encoded_pos, network_feature], dim=-1)))
        else:
            feature = self.act_layer(self.stage(encoded_pos))

        densities = torch.exp(self.density_net(feature))

        color_feature = self.act_layer(self.feature_net(feature))
        colors = torch.sigmoid(self.rgb_net(torch.cat([encoded_dir, color_feature], dim=-1)))

        attributes = {
            "xyz": xyzs,
            "dir": dirs,
            "density": densities,
            "color": colors
        }
        return attributes

@NERF_SCENE_REGISTRY.register()
def build_vanilla_scene(cfg: CfgNode) -> VanillaScene:
    """
    Build the vanilla scene defined by `cfg.MODEL.SCENE`.
    """
    return VanillaScene(
        positional_encoder=build_positional_encoder(cfg),
        directional_encoder=build_directional_encoder(cfg),
        mlp_width=cfg.MODEL.SCENE.MLP_WIDTH,
        mlp_depth=cfg.MODEL.SCENE.MLP_DEPTH,
        skip_connection=cfg.MODEL.SCENE.SKIP_CONNECTION
    )
