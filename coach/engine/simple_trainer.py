import concurrent.futures
import logging
import math
import time
from typing import Iterable

import torch
import torch.optim as torch_optim
import torch.utils.data as torch_data

from coach.utils import comm
from coach.utils.events import get_event_storage

from .base_trainer import *

__all__ = [
    "SimpleTrainer"
]

class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of training loop.
    You can refer to the implementation of this class to your own custom trainer.
    It assumes that every step, you:

    1. Compute the loss with `loss = model(data)`.
    2. Compute the gradients with `loss.backward()`.
    3. Update the model parameters with `optimizer.step()`.

    All other functions (e.g., schedulers, logging, checkpointing) are maintained
    by hooks which have been registered upon construction of the base trainer.

    Args:
        `model` (torch.nn.Module): The model to be optimized, which receives the data from the data loader
            and returns a dictionary of losses.
        `data_loader` (torch.utils.data.DataLoader | Iterable): The data loader.
        `optimizer` (torch.optim.Optimizer): The optimizer.
        `gather_metrics_freq` (int): The frequency to gather metrics from all processes to the main process.
        `zero_grad_before_step` (bool): Whether to zero gradients before each step.
        `async_write_metrics` (bool): Whether to write metrics asynchronously.
    """

    def __init__(self,
        model: torch.nn.Module,
        data_loader: torch_data.DataLoader | Iterable,
        optimizer: torch_optim.Optimizer,
        gather_metrics_freq: int = 1,
        zero_grad_before_step: bool = False,
        async_write_metrics = True
    ) -> None:
        super().__init__()

        self.model = model
        self.model.train()

        self.data_loader = iter(data_loader)
        self.optimizer = optimizer

        self.gather_metrics_freq = gather_metrics_freq
        self.zero_grad_before_step = zero_grad_before_step
        self.async_write_metrics = async_write_metrics
        self.concurrency_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.register_hooks(self.build_hooks())

    def step(self) -> None:
        assert self.model.training, "Model is not in training mode."

        step_start = time.perf_counter()
        data = next(self.data_loader)
        data_time = time.perf_counter() - step_start

        if self.zero_grad_before_step:
            self.optimizer.zero_grad()

        loss_dict: dict[str, torch.Tensor] = self.model(data)
        losses: torch.Tensor = sum(loss_dict.values())

        if not self.zero_grad_before_step:
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            self.concurrency_executor.submit(self._write_metrics, loss_dict, data_time, self.iteration)
        else:
            self._write_metrics(loss_dict, data_time, self.iteration)

        self.optimizer.step()

    def _write_metrics(
        self,
        loss_dict: dict[str, torch.Tensor],
        data_time: float,
        iteration: int,
    ) -> None:
        """
        Write metrics to logger.

        Args:
            loss_dict (dict[str, torch.Tensor]): A dictionary of scalar losses.
            data_time (float): The time spent on data loading.
            iteration (int): The current iteration.
        """
        if (self.iteration + 1) % self.gather_metrics_freq != 0:
            return

        logger = logging.getLogger(__name__)

        metric_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metric_dict["data_time"] = data_time

        storage = get_event_storage()
        storage.put_scalar("rank_data_time", data_time, curr_iter=iteration)

        # Gather metrics from all processes.
        metric_dict_reduced = comm.gather(metric_dict)

        if comm.is_main_process():
            data_time = max(x.pop("data_time") for x in metric_dict_reduced)
            storage.put_scalar("data_time", data_time, curr_iter=iteration)

            # Average the metrics.
            total_metric_dict = {
                k: sum(x[k] for x in metric_dict_reduced) / len(metric_dict_reduced)
                for k in metric_dict_reduced[0].keys()
            }
            total_loss = sum(v for k, v in total_metric_dict.items() if "loss" in k)
            if not math.isfinite(total_loss):
                logger.error(
                    "Loss is {} at iteration {}, stopping training. Loss dict = {}".format(total_loss, iteration, total_metric_dict)
                )
                raise FloatingPointError(
                    "Loss is {} at iteration {}, stopping training. Loss dict = {}".format(total_loss, iteration, total_metric_dict)
                )

            storage.put_scalar("total_loss", total_loss, curr_iter=iteration)

            if len(total_metric_dict) > 1:
                storage.put_scalars(curr_iter=iteration, **total_metric_dict)

    def state_dict(self) -> dict:
        result = super().state_dict()
        result["optimizer"] = self.optimizer.state_dict()
        return result
    
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def after_train(self) -> None:
        super().after_train()
        self.concurrency_executor.shutdown(wait=True)
