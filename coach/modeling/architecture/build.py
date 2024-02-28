import torch

from coach.config import CfgNode
from coach.utils.logger import log_api_usage
from coach.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = "Registry for the model."

def build_model(cfg: CfgNode) -> torch.nn.Module:
    """
    Build the model defined by `cfg.MODEL.NAME`.
    It does not load checkpoints from `cfg`.
    """
    model_name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    log_api_usage("modeling.architecture.{}".format(model_name))
    return model
