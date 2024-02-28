import logging

from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    MultiStepParamScheduler,
    StepWithFixedGammaParamScheduler
)
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from coach.config import CfgNode

from .scheduler import (
    LRMultiplier,
    WarmupParamScheduler
)

def build_optimizer(cfg: CfgNode, model: nn.Module) -> Optimizer:
    """Build an optimizer from given configuration."""
    params = ()

def build_scheduler(cfg: CfgNode, optimizer: Optimizer) -> LRScheduler:
    """Return a scheduler with a given configuration."""

    name = cfg.SOLVER.SCHEDULER.NAME

    if name == "MultiStepParamScheduler":
        steps = list(filter(lambda x: x <= cfg.SOLVER.MAX_ITER, cfg.SOLVER.STEPS))
        if len(steps) < len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. These values will be ignored."
            )
        scheduler = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA ** i for i in range(len(steps) + 1)],
            num_updates=cfg.SOLVER.MAX_ITER,
            milestones=steps
        )
    elif name == "CosineParamScheduler":
        end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
        assert 0.0 <= end_value <= 1.0, "end_value must be in [0.0, 1.0], got {}".format(end_value)
        scheduler = CosineParamScheduler(1.0, end_value)
    elif name == "StepWithFixedGammaParamScheduler":
        scheduler = StepWithFixedGammaParamScheduler(
            base_value=1.0,
            gamma=cfg.SOLVER.GAMMA,
            num_decays=cfg.SOLVER.NUM_DECAYS,
            num_updates=cfg.SOLVER.MAX_ITER
        )
    else:
        raise ValueError("Unknown scheduler: {}".format(name))

    scheduler = WarmupParamScheduler(
        scheduler,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
        cfg.SOLVER.WARMUP_INTERVAL
    )
    return LRMultiplier(optimizer, scheduler, max_iter=cfg.SOLVER.MAX_ITER)
