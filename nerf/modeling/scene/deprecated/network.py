from typing import Optional

import torch.nn as nn

from model.settings import build_activation_module

def build_mlp(config: dict, *, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> nn.Sequential:
    """Build the network with respect to the configuration.
    Args:
        config (dict): The configuration of the network.
        in_dim (Optional[int]): The input dimension of the network.
        out_dim (Optional[int]): The output dimension of the network.
    Returns:
        (nn.Sequential): The network.
    """

    mlp_width: int = config["mlp_width"]
    n_layers: int = config["n_layers"]
    activation: dict = config["activation"]
    last_activation: dict = config["last_activation"]

    if in_dim is None:
        in_dim = mlp_width
    if out_dim is None:
        out_dim = mlp_width

    network = []
    if n_layers == 1:
        network.append(nn.Linear(in_dim, out_dim, bias=False))
        network.append(build_activation_module(last_activation))
    else:
        for i in range(n_layers):
            if i == 0:
                network.append(nn.Linear(in_dim, mlp_width, bias=False))
                network.append(build_activation_module(activation))
            elif i == n_layers - 1:
                network.append(nn.Linear(mlp_width, out_dim, bias=False))
                network.append(build_activation_module(last_activation))
            else:
                network.append(nn.Linear(mlp_width, mlp_width, bias=False))
                network.append(build_activation_module(activation))

    return nn.Sequential(*network)
