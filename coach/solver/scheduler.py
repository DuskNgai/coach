from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    LinearParamScheduler,
    ParamScheduler
)
import torch

from coach.utils.env import TORCH_VERSION

if TORCH_VERSION < (2, 0):
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
else:
    from torch.optim.lr_scheduler import LRScheduler


class WarmupParamScheduler(CompositeParamScheduler):
    """
    Warmup a scheduler at the beginning of training.

    Args:
        `scheduler` (fvcore.common.param_scheduler.ParamScheduler): a base scheduler
            that defines the multiplier.
        `warmup_factor` (float): the multiplier for the learning rate at the first iteration.
        `warmup_length` (float): the number of iterations for warmup.
        `warmup_method` (str): method to warmup, either "constant" or "linear".
        `rescale_interval` (bool): whether to rescale scheduler interval during warmup.
    """

    def __init__(self,
        scheduler: ParamScheduler,
        warmup_factor: float,
        warmup_length: float,
        warmup_method: str = "linear",
        rescale_interval: bool = False,
    ) -> None:
        start_value = warmup_factor * scheduler(0.0)
        end_value = scheduler(0.0) if rescale_interval else scheduler(warmup_length)

        if warmup_method == "constant":
            warmup = ConstantParamScheduler(start_value)
        elif warmup_method == "linear":
            warmup = LinearParamScheduler(start_value, end_value)
        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))

        super().__init__(
            [warmup, scheduler],
            interval_scaling=["rescaled", "rescaled" if rescale_interval else "fixed"],
            lengths=[warmup_length, 1 - warmup_length],
        )


class LRMultiplier(LRScheduler):
    """
    A LRScheduler which uses fvcore `ParamScheduler` to multiply the learning rate.
    Every step, the learning rate is multiplied by the output of the scheduler.

    Args:
        `optimizer` (torch.optim.Optimizer): optimizer to be scheduled.
        `scheduler` (fvcore.common.param_scheduler.ParamScheduler): a base scheduler
            that defines the multiplier.
        `max_iter` (int): the number of iterations.
        `last_iter` (int): the index of the last iteration. Default: -1.
    """

    def __init__(self,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        max_iter: int,
        last_iter: int = -1
    ) -> None:
        if not isinstance(scheduler, ParamScheduler):
            raise ValueError("Scheduler must be an instance of fvcore ParamScheduler.")
        self._scheduler = scheduler
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def get_lr(self) -> list[float]:
        multiplier = self._scheduler(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def state_dict(self) -> dict:
        return {
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }
