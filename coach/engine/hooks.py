from collections import Counter 
import datetime
import logging
import time

from fvcore.common.checkpoint import (
    Checkpointer,
    PeriodicCheckpointer as _PeriodicCheckpointer
)
from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.timer import Timer
import torch

from coach.utils.events import EventWriter
from coach.solver.scheduler import LRMultiplier

__all__ = [
    "HookBase",
    "IterationTimer",
    "LRScheduler",
    "PeriodicCheckpointer",
    "PeriodicWriter",
]


class HookBase(object):
    """
    Base class for hooks that can be registered with `TrainerBase`.
    
    >>> hook.before_train()
    >>> for iter in range(start_iter, max_iter):
    >>>     hook.before_step()
    >>>     trainer.step()
    >>>     hook.after_step()
    >>> hook.after_train()

    Notes:
        1. There is a weak reference to the trainer object, so you can access
           the trainer via `self.trainer()`.
        2. If there is something that can be done either in `before_step` or in `after_step`,
           always prefer `after_step` as it `before_step` should only take negligible amount of time.
           Following this convention will allow hooks that do care about the difference between
           `before_step` and `after_step` (e.g. timer) to work properly.
    """

    # A weak reference to the trainer object.
    trainer = None

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def after_backward(self):
        """
        Called after the backward pass of each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default.
        By overriding `state_dict` and `load_state_dict`,
        hooks can be made checkpointable.
        """
        return {}


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each `step` call in the trainer).

    This hook should be placed at the beginning of the hook list to obtain accurate timing.
    """

    def __init__(self, warmup_iter: int = 3):
        self._warmup_iter = warmup_iter
        self._start_time = 0.0
        self._step_timer = Timer()
        self._total_timer = Timer()

    def before_train(self):
        # Start the total training timer
        self._start_time = time.perf_counter()
        # Stop the total timer until the first step is executed
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.storage.iteration + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        # Reset the step timer to record the time for this iteration
        self._step_timer.reset()
        # Resume the total timer to exclude the time for the hooks
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step, the current step is done but not yet counted
        iter_done = self.trainer.storage.iteration - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            # Read the timer
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        # Pause the total timer until the next step is executed
        self._total_timer.pause()

class LRScheduler(HookBase):
    """
    Wrapper for PyTorch LR scheduler.
    It is executed after each optimizer step.
    If arguments are not specified, it will be obtained from the trainer.

    Args:
        `optimizer` (torch.optim.Optimizer): an Optimizer.
        `scheduler` (torch.optim.lr_scheduler._LRScheduler): a PyTorch LR scheduler.
    """

    def __init__(self, optimizer=None, scheduler=None):
        self._optimizer = optimizer
        self._scheduler = scheduler

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        if self._scheduler is None:
            self._scheduler = self.trainer.scheduler
        return self._scheduler

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = self.trainer.optimizer
        return self._optimizer

    @staticmethod
    def get_best_param_group_id(optimizer: torch.optim.Optimizer):
        """
        Since there may be different lrs for different param groups,
        we need to find the param group with the largest number of params.
        And then mark out the most common lr in this group.
        """
        largest_param_group = max(len(group["params"]) for group in optimizer.param_groups)

        if largest_param_group == 1:
            lr_count = Counter(group["lr"] for group in optimizer.param_groups)
            # Get the most common lr
            lr = lr_count.most_common(1)[0][0]
            for i, group in enumerate(optimizer.param_groups):
                if group["lr"] == lr:
                    return i
        else:
            for i, group in enumerate(optimizer.param_groups):
                if len(group["params"]) == largest_param_group:
                    return i

    def before_train(self):
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self.optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iteration - 1,
            )
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self.optimizer)

    def after_step(self):
        # Record the current step lr and scheduler lr.
        lr = self.optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()

    def state_dict(self) -> dict:
        if isinstance(self, torch.optim.lr_scheduler.LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict):
        if isinstance(self, torch.optim.lr_scheduler.LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state dict ...")
            self.scheduler.load_state_dict(state_dict)

class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    """
    Same as fvcore's PeriodicCheckpointer, but is also a hook.
    Save checkpoints periodically.
    """

    def __init__(self, checkpointer: Checkpointer, period: int):
        super().__init__(checkpointer, period)

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        self.step(self.trainer.iteration)

class PeriodicWriter(HookBase):
    """
    Write events periodically to the EventStorage.
    """

    def __init__(self, writers: list[EventWriter], period: int):
        self._writers = writers
        for writer in writers:
            assert isinstance(writer, EventWriter), "Writers must be a list of EventWriter."
        self._period = period

    def after_step(self):
        iter = self.trainer.iteration + 1
        if iter % self._period == 0 or iter == self.trainer.max_iter:
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            writer.write()
            writer.close()
