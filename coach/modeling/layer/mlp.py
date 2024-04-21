from typing import Optional

import torch
import torch.nn as nn

__all__ = ["Mlp"]


class Mlp(nn.Module):
    """
    Mlp without last activation function.
    """
    def __init__(self,
        in_channels: int,
        hidden_layers: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        bias: bool = False,
        act_layer: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or hidden_channels

        if isinstance(hidden_channels, list):
            assert len(hidden_channels) == hidden_layers - 1, f"The length of hidden_channels ({hidden_channels}) should be hidden_layers - 1 ({hidden_layers - 1})."
        elif isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (hidden_layers - 1)

        layers = []
        if hidden_layers == 0:
            layers.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            layers.append(nn.Linear(in_channels, hidden_channels[0], bias=bias))
            layers.append(act_layer())
            for i in range(1, hidden_layers - 1):
                layers.append(nn.Linear(hidden_channels[i - 1], hidden_channels[i], bias=bias))
                layers.append(act_layer())
            layers.append(nn.Linear(hidden_channels[-1], out_channels, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
