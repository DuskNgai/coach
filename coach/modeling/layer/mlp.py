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

        layers = []
        if hidden_layers == 0:
            layers.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            layers.append(nn.Linear(in_channels, hidden_channels, bias=bias))
            layers.append(act_layer())
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                layers.append(act_layer())
            layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
