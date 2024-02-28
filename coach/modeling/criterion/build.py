from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

from .criterion import Criterion

CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = "Registry for the model."

def build_criterion(cfg: CfgNode) -> Criterion:
    """
    Build the criterion defined by `cfg.CRITERION.NAME`.
    It does not load checkpoints from `cfg`.
    """
    criterion_name = cfg.MODEL.CRITERION.NAME
    criterion = CRITERION_REGISTRY.get(criterion_name)(cfg)
    log_api_usage("modeling.criterion.{}".format(criterion_name))
    return criterion
