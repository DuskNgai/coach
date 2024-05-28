import torch
from fvcore.common.registry import Registry

from coach.config import CfgNode
from coach.utils.logger import log_api_usage

CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = "Registry for the model."

def build_criterion(cfg: CfgNode) -> torch.nn.Module:
    """
    Build the criterion defined by `cfg.MODEL.CRITERION.NAME`.
    """
    criterion_name = cfg.MODEL.CRITERION.NAME
    criterion = CRITERION_REGISTRY.get(criterion_name)(cfg)
    log_api_usage("modeling.criterion.{}".format(criterion_name))
    return criterion
