import logging
import weakref

from coach.engine.hooks import (
    HookBase,
    IterationTimer,
    LRScheduler,
    PeriodicCheckpointer,
    PeriodicWriter
)
from coach.utils.events import EventStorage
from coach.utils.logger import log_api_usage
from coach.utils import comm


__all__ = [
    "TrainerBase",
]


class TrainerBase(object):
    """
    Base class for iterative trainer with hooks.

    Attributes:
        `hooks` (list[HookBase]): list of hooks to be registered.
        `iteration` (int): the current iteration number.
        `start_iter` (int): the iteration number to start with.
        `max_iter` (int): the iteration number to end training.
        `storage` (EventStorage): the event storage that is used during training.
    """

    def __init__(self) -> None:
        self._hooks: list[HookBase] = []
        self.iteration: int = 0
        self.start_iter: int = 0
        self.max_iter: int = 0
        self.storage: EventStorage = None
        log_api_usage("trainer.{}".format(self.__class__.__name__))

    def register_hooks(self, hooks_for_trainer: list[HookBase]) -> None:
        """
        Register hooks with the trainer.
        They are executed in the order of registration.

        Args:
            hooks_for_trainer (list[HookBase]): list of hooks to be registered.
        """

        hooks_for_trainer = list(filter(None, hooks_for_trainer))
        for h in hooks_for_trainer:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks_for_trainer)

    def build_hooks(self) -> list[HookBase]:
        result = [
            IterationTimer(),
            LRScheduler()
        ]

        # Do PreciseBN before checkpointer,
        # because it updates the model and need to be saved by checkpointer.
        if comm.is_main_process():
            result.append(PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD))

        if comm.is_main_process():
            result.append(PeriodicWriter(self.build_writers(), period=self.cfg.SOLVER.LOG_PERIOD))

        return result

    def train(self, start_iter: int, max_iter: int) -> None:
        """
        Args:
            `start_iter` (int): the iteration number to start with.
            `max_iter` (int): the iteration number to end training.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.start_iter = start_iter
        self.max_iter = max_iter
        self.iteration = self.start_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iteration in range(start_iter, max_iter):
                    self.before_step()
                    self.step()
                    self.after_step()
                # See `is_loop_completed`
                self.iteration += 1
            except Exception as e:
                logger.exception("Exception during training:")
                raise Exception from e
            finally:
                self.after_train()

    def is_loop_completed(self) -> bool:
        """
        `self.iter == max_iter` can be used to check if the training loop
        completed successfully by `after_train`.
        This function is used in hooks to check if the training loop.
        """
        return self.iteration == self.max_iter

    def before_train(self) -> None:
        for h in self._hooks:
            h.before_train()

    def after_train(self) -> None:
        self.storage.iteration = self.iteration
        for h in self._hooks:
            h.after_train()

    def before_step(self) -> None:
        # Maintain the invariant that `storage.iter == trainer.iter`
        self.storage.iteration = self.iteration
        for h in self._hooks:
            h.before_step()

    def after_step(self) -> None:
        for h in self._hooks:
            h.after_step()

    def after_backward(self) -> None:
        for h in self._hooks:
            h.after_backward()

    def step(self) -> None:
        raise NotImplementedError()

    def state_dict(self) -> dict:
        result = {
            "iteration": self.iteration
        }
        hook_state = {}
        for h in self._hooks:
            state = h.state_dict()
            if state:
                name = type(h).__qualname__
                hook_state[name] = state
        if hook_state:
            result["hooks"] = hook_state
        return result
    
    def load_state_dict(self, state_dict: dict) -> None:
        logger = logging.getLogger(__name__)
        self.iteration = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                if type(h).__qualname__ == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning("No hook of type {} found, its state_dict is ignored.".format(key))
