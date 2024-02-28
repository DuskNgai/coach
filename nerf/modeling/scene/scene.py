from abc import ABCMeta, abstractmethod

import torch.nn as nn

__all__ = ["Scene"]

class Scene(nn.Module, metaclass=ABCMeta):
    """
    Abstract class for scene, i.e. the 3D world.
    It can be neural networks or grid-based representations.
    It receives the sampled points from the sampler and outputs the attributes of the points.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self) -> None:
        """
        Sample points from the scene.
        """
        raise NotImplementedError
