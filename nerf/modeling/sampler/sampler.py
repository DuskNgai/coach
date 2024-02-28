from abc import ABCMeta, abstractmethod

import torch.nn as nn

__all__ = ["Sampler"]

class Sampler(nn.Module, metaclass=ABCMeta):
    """
    Abstract class for sampler which provide samples (queries) for scene.

    Sampler receives a batch of poses and intrinsics, and returns a batch of queries.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self) -> None:
        pass
