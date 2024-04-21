import functools
from typing import TypeVar

import numpy as np
import torch
import torch.distributed as dist

def get_world_size() -> int:
    """
    Returns:
        (int): Returns the number of processes in the current process group.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Returns:
        (int): Returns the rank of current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """
    Returns:
        (bool): True if the current process is the main process.
    """
    return get_rank() == 0


_LOCAL_PROCESS_GROUP = None
_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use coach's `launch()` "
    "to start processes and initialize pytorch process group. If you need to start "
    "processes in other ways, please call `comm.create_local_process_group("
    "num_workers_per_machine)` after calling `torch.distributed.init_process_group()`."
)


@functools.lru_cache()
def create_local_process_group(num_gpus_per_machine: int) -> None:
    """
    Create a local process group for processes on the same machine.
    Before calling functions in `comm`, you need to first call this function.

    Args:
        `num_gpus_per_machine` (int): The number of GPUs per machine.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None, "Local process group already created!"
    assert get_world_size() % num_gpus_per_machine == 0

    num_machines = get_world_size() // num_gpus_per_machine
    machine_rank = get_rank() // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group() -> dist.ProcessGroup:
    """
    Returns:
        (dist.ProcessGroup): The local process group, a.k.a. the process on current (local) machine.
    """
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return _LOCAL_PROCESS_GROUP


def get_local_size() -> int:
    """
    Returns:
        (int): Returns the number of processes on the local process group.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_local_rank() -> int:
    """
    Returns:
        (int): Returns the rank of current process on the local process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def synchronize() -> None:
    """
    Synchronize all processes.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if get_world_size() == 1:
        return

    if dist.get_backend() == dist.Backend.NCCL:
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group() -> dist.ProcessGroup:
    """
    Returns:
        (dist.ProcessGroup): The global Gloo process group.
    """
    if dist.get_backend() == dist.Backend.NCCL:
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


T = TypeVar('T')


def all_gather(data: T, group=None) -> list[T]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    All gather means gather data from all processes to all processes.

    Args:
        `data` (Any): Any picklable object.
        `group` (dist.ProcessGroup): The process group to gather results from.
            By default, a group containing all ranks on gloo backend.

    Returns:
        (list[Any]): A list containing all data from each rank.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()

    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    
    buffer = [None for _ in range(world_size)]
    dist.all_gather_object(buffer, data, group=group)
    return buffer


def gather(data: T, dst: int = 0, group=None) -> list[T]:
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Gather means gather data from all processes to one process.

    Args:
        `data` (Any): Any picklable object.
        `dst` (int): The rank of the destination process, default to main process.
        `group` (dist.ProcessGroup): The process group to gather results from.
            By default, a group containing all ranks on gloo backend.

    Returns:
        (list[Any]): A list containing all data from each rank.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()

    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]

    rank = get_rank(group=group)
    if rank == dst:
        buffer = [None for _ in range(world_size)]
        dist.gather_object(data, buffer, dst=dst, group=group)
        return buffer
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        (int): a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]
