from datetime import timedelta
import logging
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from coach.utils import comm

__all__ = [
    "DEFAULT_TIMEOUT",
    "launch"
]


DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port() -> int:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func: Callable[..., Any],
    num_gpus_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = None,
    args: tuple = (),
    timeout=DEFAULT_TIMEOUT,
) -> None:
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by `num_gpus_per_machine`) on each machine.

    Args:
        `main_func` (Callable[..., Any]): A function that is called on all processes after spawning.
        `num_gpus_per_machine` (int): Number of GPUs per machine.
        `num_machines` (int): Number of machines for distributed training.
        `machine_rank` (int): The rank of this machine (one per machine).
        `dist_url` (str): URL to connect to for distributed training, including protocol.
        `args` (tuple): The arguments passed into `main_func`.
        `timeout` (timedelta): Timeout for distributed workers.
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = "tcp://127.0.0.1:{}".format(port)

        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.start_processes(
            _distributed_worker,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            nprocs=num_gpus_per_machine,
            daemon=False,
        )

    else:
        main_func(*args)


def _distributed_worker(
    local_rank: int,
    main_func: Callable[..., Any],
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: tuple,
    timeout: timedelta
) -> None:
    """
    Initialize the distributed environment and then call `main_func` on the current process.

    Args:
        See `launch`.
    """
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        assert num_gpus_per_machine <= torch.cuda.device_count()

    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL" if has_gpu else "GLOO",
            init_method=dist_url,
            timeout=timeout,
            world_size=world_size,
            rank=global_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group.
    comm.create_local_process_group(num_gpus_per_machine)
    if has_gpu:
        torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling `init_process_group`
    comm.synchronize()

    main_func(*args)
