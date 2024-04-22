from abc import ABCMeta, abstractmethod
from typing import Any

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

    def inference(self, *args) -> Any:
        if len(args) == 1:
            return args[0]
        return args
