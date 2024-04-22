from typing import List, Optional, Union

import torch
import torch.nn as nn

__all__ = ["ConvNet"]


class ConvNet(nn.Module):
    """
    Multi-layer convolutional neural network.
    Downsample the input tensor by stride = 2.
    """
    def __init__(self,
        in_channels: int,
        hidden_layers: int,
        hidden_channels: Optional[Union[int, List[int]]] = None,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channels, hidden_channels[0], kernel_size, stride, padding, bias=bias))
            layers.append(act_layer())
            layers.append(nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size, 2, padding, bias=bias))
            layers.append(act_layer())
            for i in range(1, hidden_layers - 1):
                layers.append(nn.Conv2d(hidden_channels[i - 1], hidden_channels[i], kernel_size, stride, padding, bias=bias))
                layers.append(act_layer())
                layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i], kernel_size, 2, padding, bias=bias))
                layers.append(act_layer())
            layers.append(nn.Conv2d(hidden_channels[-1], out_channels, kernel_size, stride, padding, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
