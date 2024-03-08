from collections import defaultdict
import datetime
from functools import cached_property
import json
import logging
from numbers import Number
import os
import time

from fvcore.common.history_buffer import HistoryBuffer
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .file_io import PathManagerSingleton

__all__ = [
    "has_event_storage",
    "get_event_storage",
    "EventStorage",
    "CommonMetricPrinter",
    "JSONWriter",
    "TensorboardXWriter",
]


class EventStorage(object):
    """
    The user-facing class that provides metric storage functionalities.
    Currently, we support scalars and images.

    Args:
        `start_iter` (int): the iteration number to start with.
    """

    def __init__(self, start_iter: int = 0) -> None:
        self._iter = start_iter
        self._prefix = ""
        self._image_data: list[tuple[str, np.ndarray | torch.Tensor, int]] = []
        self._history_buffers = defaultdict(HistoryBuffer)
        self._latest_scalars: dict[str, tuple[Number, int]] = {}
        self._smoothing_hints: dict[str, bool] = {}

    def put_image(self, image_name: str, image: np.ndarray | torch.Tensor) -> None:
        """
        Add an image to the event storage that will be shown in tensorboard.

        Args:
            `image_name` (str): the name of the image.
            `image_tensor` (torch.Tensor): the image tensor.
                The image should be in shape (C, H, W) where C = 3.
                The data type should be uint8 in range [0, 255] or float in range [0, 1].
        """
        self._image_data.append((image_name, image, self.iteration))

    def put_scalar(self, name: str, value: Number, smoothing_hint: bool = True, curr_iter: int = None) -> None:
        """
        Add a scalar to the event storage with name.

        Args:
            `name` (str): the name of the scalar.
            `value` (Number): the value of the scalar.
            `smoothing_hint` (bool): whether the scalar is smoothed when logged.
            `curr_iter` (int): the explicit iteration number to put the scalar.
        """
        name = self._prefix + name
        value = float(value)
        curr_iter = self.iteration if curr_iter is None else curr_iter

        history = self._history_buffers[name]
        history.update(value, curr_iter)
        self._latest_scalars[name] = (value, curr_iter)

        hint = self._smoothing_hints.get(name)
        if hint is None:
            self._smoothing_hints[name] = smoothing_hint
        else:
            assert hint == smoothing_hint, "Scalar {} already has a smoothing hint!".format(name)

    def put_scalars(self, *, smoothing_hint: bool = True, curr_iter: int = None, **kwargs) -> None:
        """
        Add multiple scalars to the event storage.

        Args:
            `smoothing_hint` (bool): whether the scalar is smoothed when logged.
            `curr_iter` (int): the explicit iteration number to put the scalar.
            `**kwargs` (dict): the scalars to put.
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint, curr_iter)

    def history(self, name: str) -> HistoryBuffer:
        """
        Returns:
            the `HistoryBuffer` object that stores the history of the given scalar.
        """
        h = self._history_buffers.get(name)
        if h is None:
            raise KeyError("No history of {} found.".format(name))
        return h

    def histories(self) -> dict[str, HistoryBuffer]:
        """
        Returns:
            a dict that maps names to `HistoryBuffer` objects.
        """
        return self._history_buffers
    
    def latest(self) -> dict[str, tuple[Number, int]]:
        """
        Returns:
            a dict that maps names to (value, iteration) tuples representing the latest scalars.
        """
        return self._latest_scalars

    def smoothing_hints(self) -> dict[str, bool]:
        """
        Returns:
            a dict that maps names to boolean indicating whether the data is smoothed.
        """
        return self._smoothing_hints

    def count_samples(self, name: str, window_size: int = 20) -> int:
        """
        Returns:
            the number of samples logged in history buffer if `window_size` is longer than the history.
        """
        n_samples = 0
        data = self._history_buffers[name].values()
        for _, iteration in reversed(data):
            if iteration > data[-1][1] - window_size:
                n_samples += 1
            else:
                break
        return n_samples

    def latest_with_smoothing_hint(self, window_size: int = 20) -> dict[str, tuple[Number, int]]:
        """
        Returns:
            a dict that maps names to (value, iteration) tuples representing the latest scalars.
            If the scalar is marked as smoothed, all the scalars in the window will be smoothed.
        """
        result = {}
        for name, (value, iteration) in self.latest().items():
            if self._smoothing_hints[name]:
                n_samples = self.count_samples(name, window_size)
                result[name] = (self.history(name).median(n_samples), iteration)
            else:
                result[name] = (value, iteration)
        return result

    def step(self) -> None:
        """
        Step the iteration at each iteration.
        """
        self._iter += 1

    @property
    def iteration(self):
        return self._iter

    @iteration.setter
    def iteration(self, i: int) -> None:
        self._iter = i

    def clear_images(self) -> None:
        """
        Clear the image data.
        This method should be called after each tensorboard logging.
        """
        self._image_data = []

    def __enter__(self):
        _EVENT_STORAGE_STACK.append(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _EVENT_STORAGE_STACK[-1] is self, "Nesting violation!"
        _EVENT_STORAGE_STACK.pop()


_EVENT_STORAGE_STACK: list[EventStorage] = []


def has_event_storage() -> bool:
    """
    Whether there is an `EventStorage` in the stack.
    """
    return len(_EVENT_STORAGE_STACK) > 0


def get_event_storage() -> EventStorage:
    """
    Get the topmost `EventStorage` (currenly being used) in the stack.
    """
    assert has_event_storage(), "`get_event_storage()` has to be called inside a 'with EventStorage(...)' context!"
    return _EVENT_STORAGE_STACK[-1]


class EventWriter(object):
    """
    Base class for writers that can write events to EventStorage.
    """

    def write(self):
        """
        Write the events in the EventStorage to file.
        """
        raise NotImplementedError

    def close(self):
        """
        Close the writer.
        """
        pass


class CommonMetricPrinter(EventWriter):
    """
    During training, the main loop often wants to print some metrics to stdout.
    This class provides a simple way to achieve that.
    It prints eta, losses, metrics, times, lr and memory.

    Args:
        `max_iter` (int): the maximum number of iterations.
        `window_size` (int): the window size to smooth scalars.
    """
    def __init__(self, max_iter: int = None, window_size: int = 20):
        self.logger = logging.getLogger("coach.utils.events")
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write_time = None

    def _get_eta(self, storage: EventStorage) -> str | None:
        if self._max_iter is None:
            return None

        iteration = storage.iteration
        try:
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint = False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            eta = None
            if self._last_write_time is not None:
                estimated_iter_time = (time.perf_counter() - self._last_write_time[1]) / (iteration - self._last_write_time[0])
                eta_seconds = estimated_iter_time * (self._max_iter - iteration - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write_time = (iteration, time.perf_counter())
            return eta

    def write(self):
        storage = get_event_storage()
        iteration = storage.iteration

        # No data logged if training process is ended.
        if iteration == self._max_iter:
            return

        try:
            avg_data_time = storage.history("data_time").median(
                storage.count_samples("data_time", self._window_size)
            )
            last_data_time = storage.history("data_time").latest()
        except KeyError:
            avg_data_time = None
            last_data_time = None

        try:
            avg_time = storage.history("time").global_avg()
            last_time = storage.history("time").latest()
        except KeyError:
            avg_time = None
            last_time = None

        try:
            lr = storage.history("lr").latest()
        except KeyError:
            lr = None

        eta = self._get_eta(storage)

        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_memory = None

        self.logger.info(
            "{eta}\titer: {iteration}\t{losses}\t{metrics}\t{time}\t{data_time}\t{lr}\t{memory}".format(
                eta="eta: {}".format(eta) if eta is not None else "",
                iteration=iteration,
                losses="  ".join([
                    "{}: {:.4f}".format(name, value.median(storage.count_samples(name, self._window_size)))
                    for name, value in storage.histories().items() if "loss" in name
                ]),
                metrics="  ".join([
                    "{}: {:.4f}".format(name, value.median(storage.count_samples(name, self._window_size)))
                    for name, value in storage.histories().items() if "metric" not in name
                ]),
                time="avg time: {:.4f} (last: {:.4f})".format(avg_time, last_time) if avg_time is not None else "",
                data_time="avg data time: {:.4f} (last: {:.4f})".format(avg_data_time, last_data_time) if avg_data_time is not None else "",
                lr="{:.4e}".format(lr) if lr is not None else "N/A",
                memory="max memory: {:.4f} MB".format(max_memory) if max_memory is not None else "",
            )
        )


class JSONWriter(EventWriter):
    """
    Write scalars to a json file.
    For each log iteration, the scalars will be stored in json file in one line,
    which is easier to parse.

    Args:
        `file_path` (str): the json path to store the scalars.
        `window_size` (int): the window size to smooth scalars.
    """
    def __init__(self, file_path: str, window_size: int = 20) -> None:
        self._file_handle = PathManagerSingleton.open(file_path, "a")
        self._window_size = window_size
        self._last_write = -1

    def write(self) -> None:
        storage = get_event_storage()
        to_save = defaultdict(dict)

        # Select scalars to save.
        for name, (value, iteration) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iteration > self._last_write:
                to_save[iteration][name] = value

        # Update the last write time by the max iteration in this write.
        if len(to_save) > 0:
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        # Write scalars to json file.
        for iteration, named_scalar in to_save.items():
            named_scalar["iteration"] = iteration
            self._file_handle.write(json.dumps(named_scalar, sort_keys=True) + "\n")
        self._file_handle.flush()

        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self) -> None:
        self._file_handle.close()


class TensorboardXWriter(EventWriter):
    """
    Write scalars and images to tensorboard event file.

    Args:
        `file_path` (str): the json path to store the scalars.
        `window_size` (int): the window size to smooth scalars.
        `**kwargs` (dict): the arguments to pass to `SummaryWriter`.
    """
    def __init__(self, file_path: str, window_size: int = 20, **kwargs) -> None:
        self._writer_args = {"log_dir": file_path, **kwargs}
        self._window_size = window_size
        self._last_write = -1

    @cached_property
    def _writer(self) -> SummaryWriter:
        return SummaryWriter(**self._writer_args)

    def write(self) -> None:
        storage = get_event_storage()

        # Select scalars to save.
        new_last_write = self._last_write
        for name, (value, iteration) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iteration > self._last_write:
                self._writer.add_scalar(name, value, iteration)
                new_last_write = max(new_last_write, iteration)
        self._last_write = new_last_write

        # Select images to save.
        if len(storage._image_data) > 0:
            for image_name, image, iteration in storage._image_data:
                self._writer.add_image(image_name, image, iteration)
            # Storage assumes only this writer will call `clear_images()`.
            storage.clear_images()

    def close(self) -> None:
        self._writer.close()
