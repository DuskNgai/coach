import atexit
import functools
import logging
import os
import sys
from typing import IO, Optional, Union

from termcolor import colored
import torch

from .file_io import PathManagerSingleton

__all__ = [
    "setup_logger",
]


class _ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs) -> None:
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if self._abbrev_name != "":
            self._abbrev_name += "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

@functools.lru_cache()
def setup_logger(
    output: Optional[str] = None,
    distributed_rank: int = 0,
    *,
    color: bool = True,
    name: str = "Coach",
    abbrev_name: Optional[str] = None,
    enable_propagation: bool = False,
    configure_stdout: bool = True,
) -> logging.Logger:
    """
    Initialize the Coach logger and set its verbosity level to "DEBUG".

    Args:
        `output` (str): a file name or a directory to save log.
            If set to None, will not log to any file.
        `distributed_rank` (int): the rank of the process.
        `name` (str): the root module name of the logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = enable_propagation

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout logger: master process only
    if configure_stdout and distributed_rank == 0:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        if color:
            formatter = _ColoredFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=abbrev_name,
            )
        else:
            formatter = plain_formatter
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logger: all processes
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManagerSingleton.mkdirs(os.path.dirname(filename))

        handler = logging.StreamHandler(_cached_log_stream(filename))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(plain_formatter)
        logger.addHandler(handler)

    return logger

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename: str) -> Union[IO[bytes], IO[str]]:
    """
    Cache the opened file object, so that different calls to `setup_logger`
    with the same filename will write to the same file object.
    """
    io = PathManagerSingleton.open(filename, "a")
    atexit.register(io.close)
    return io

def log_api_usage(identifier: str) -> None:
    """
    This function is used to record the usage of the PyTorch API.
    If the environment variable `PYTORCH_API_USAGE_STDERR=1` is set,
    you can see the PyTorch API usage log on stderr.
    """
    torch._C._log_api_usage_once("coach.{}".format(identifier))
