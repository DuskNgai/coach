from abc import ABCMeta, abstractmethod

import torch.nn as nn

__all__ = ["Encoder"]


class Encoder(nn.Module, metaclass=ABCMeta):
    """
    The base class for all encoder, which maps low-dimentional input into high-dimentional feature.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def out_channels(self) -> int:
        """
        The number of channels of the output feature.
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """
        Encode the low dimentional input into high dimentional feature.

        Args:
            tensor_in (torch.Tensor): [..., in_dim], low dimentional input.
        Returns:
            (torch.Tensor): [..., out_dim], high dimentional feature.
        """
        pass
