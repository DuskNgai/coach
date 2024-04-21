from abc import ABCMeta, abstractmethod

import torch.nn as nn

class Criterion(nn.Module, metaclass=ABCMeta):
    """
    Base class for custom criterion.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self) -> None:
        raise NotImplementedError
