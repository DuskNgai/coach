import torch.nn as nn

from coach.config import CfgNode

from .build import CRITERION_REGISTRY
from .criterion import Criterion

@CRITERION_REGISTRY.register()
def build_pytorch_criterion(cfg: CfgNode) -> Criterion:
    """
    Build the criterion defined by `cfg.CRITERION.NAME`.
    """
    loss_type = cfg.MODEL.CRITERION.TYPE
    if loss_type == "MSELoss":
        criterion = nn.MSELoss()
    elif loss_type == "L1Loss":
        criterion = nn.L1Loss()
    elif loss_type == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "HuberLoss":
        criterion = nn.HuberLoss()
    else:
        raise KeyError("Unknown loss type: {}".format(loss_type))

    return criterion
