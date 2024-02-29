from collections import defaultdict
import copy
from enum import Enum
import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

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
from coach.utils.env import TORCH_VERSION

from .scheduler import (
    LRMultiplier,
    WarmupParamScheduler
)

class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]

def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip

def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip

def build_optimizer(cfg: CfgNode, model: nn.Module) -> Optimizer:
    """Build an optimizer from given configuration."""
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
    )
    sgd_args = {
        "params": params,
        "lr": cfg.SOLVER.BASE_LR,
        "momentum": cfg.SOLVER.MOMENTUM,
        "nesterov": cfg.SOLVER.NESTEROV,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }
    if TORCH_VERSION >= (1, 12):
        sgd_args["foreach"] = True
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD(**sgd_args))

def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    lr_factor_func: Optional[Callable] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        lr_factor_func: function to calculate lr decay rate by mapping the parameter names to
            corresponding lr decay rate. Note that setting this option requires
            also setting ``base_lr``.
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.
    """
    if overrides is None:
        overrides = {}

    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    if lr_factor_func is not None:
        if base_lr is None:
            raise ValueError("lr_factor_func requires base_lr")

    norm_module_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            if lr_factor_func is not None:
                hyperparams["lr"] *= lr_factor_func(f"{module_name}.{module_param_name}")

            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return reduce_param_groups(params)

def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Transform parameter groups into per-parameter structure.
    # Later items in `params` can overwrite parameters set in previous items.
    ret = defaultdict(dict)
    for item in params:
        assert "params" in item
        cur_params = {x: y for x, y in item.items() if x != "params" and x != "param_names"}
        if "param_names" in item:
            for param_name, param in zip(item["param_names"], item["params"]):
                ret[param].update({"param_names": [param_name], "params": [param], **cur_params})
        else:
            for param in item["params"]:
                ret[param].update({"params": [param], **cur_params})
    return list(ret.values())

def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reorganize the parameter groups and merge duplicated groups.
    # The number of parameter groups needs to be as small as possible in order
    # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    # of using a parameter_group per single parameter, we reorganize the
    # parameter groups and merge duplicated groups. This approach speeds
    # up multi-tensor optimizer significantly.
    params = _expand_param_groups(params)
    groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != "params" and x != "param_names")
        groups[cur_params].append({"params": item["params"]})
        if "param_names" in item:
            groups[cur_params][-1]["param_names"] = item["param_names"]

    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = list(
            itertools.chain.from_iterable([params["params"] for params in param_values])
        )
        if len(param_values) > 0 and "param_names" in param_values[0]:
            cur["param_names"] = list(
                itertools.chain.from_iterable([params["param_names"] for params in param_values])
            )
        ret.append(cur)
    return ret

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
