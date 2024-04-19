import time
from typing import Iterable

import torch
import torch.optim as torch_optim
import torch.utils.data as torch_data

from coach.utils.events import get_event_storage

from .simple_trainer import SimpleTrainer

__all__ = [
    "AMPTrainer"
]


class AMPTrainer(SimpleTrainer):
    """
    Inherit from SimpleTrainer and add support for automatic mixed precision (AMP).
    """

    def __init__(self,
        model: torch.nn.Module,
        data_loader: torch_data.DataLoader | Iterable,
        optimizer: torch_optim.Optimizer,
        gather_metrics_freq: int = 1,
        zero_grad_before_step: bool = False,
        async_write_metrics = True,
        grad_scaler: torch.cuda.amp.GradScaler | None = None,
        precision: torch.dtype = torch.float16,
        log_grad_scalar: bool = False
    ) -> None:
        UNSUPPORTED = "AMPTrainer does not support simgle-process multi-device training."
        if isinstance(model, torch.nn.DataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), UNSUPPORTED
        assert not isinstance(model, torch.nn.parallel.DataParallel), UNSUPPORTED

        assert torch.cuda.is_available(), "AMPTrainer only supports CUDA training."

        super().__init__(
            model,
            data_loader,
            optimizer,
            gather_metrics_freq,
            zero_grad_before_step,
            async_write_metrics
        )

        if grad_scaler is None:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.grad_scaler = grad_scaler

        self.precision = precision
        self.log_grad_scalar = log_grad_scalar

    def step(self) -> None:
        assert self.model.training, "Model is not in training mode."

        step_start = time.perf_counter()
        data = next(self.data_loader)
        data_time = time.perf_counter() - step_start

        if self.zero_grad_before_step:
            self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=self.precision):
            loss_dict: dict[str, torch.Tensor] = self.model(data)
            losses: torch.Tensor = sum(loss_dict.values())

        if not self.zero_grad_before_step:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scalar:
            storage = get_event_storage()
            storage.put_scalar("grad_scalar", self.grad_scaler.get_scale(), smoothing_hint=False)

        self.after_backward()

        if self.async_write_metrics:
            self.concurrency_executor.submit(self._write_metrics, loss_dict, data_time, self.iteration)
        else:
            self._write_metrics(loss_dict, data_time, self.iteration)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self) -> dict:
        result = super().state_dict()
        result["grad_scaler"] = self.grad_scaler.state_dict()
        return result
    
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict("grad_scaler"))
