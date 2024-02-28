from abc import ABCMeta, abstractmethod

import torch.nn as nn

__all__ = ["Renderer"]

class Renderer(nn.Module, metaclass=ABCMeta):
    """
    Composite the attributes to obtain the final rendering result (color, depth ...).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self) -> None:
        pass
