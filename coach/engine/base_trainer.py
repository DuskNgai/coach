import logging
import weakref

from coach.utils.events import EventStorage
from coach.utils.logger import log_api_usage

__all__ = [
    "HookBase",
    "TrainerBase",
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

    def register_hooks(self, hooks: list[HookBase]) -> None:
        """
        Register hooks with the trainer.
        They are executed in the order of registration.

        Args:
            hooks (list[HookBase]): list of hooks to be registered.
        """

        hooks = list(filter(None, hooks))
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

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
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def is_loop_completed(self):
        """
        `self.iter == max_iter` can be used to check if the training loop
        completed successfully by `after_train`.
        This function is used in hooks to check if the training loop.
        """
        return self.iteration == self.max_iter

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iteration = self.iteration
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that `storage.iter == trainer.iter`
        self.storage.iteration = self.iteration
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def after_backward(self):
        for h in self._hooks:
            h.after_backward()

    def step(self):
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
    
    def load_state_dict(self, state_dict: dict):
        logger = logging.getLogger(__name__)
        self.iteration = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                if type(h).__qualname__ == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning("No hook of type {} found, its state_dict is ignored.".format(key))
