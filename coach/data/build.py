import logging
from typing import Any

import torch
import torch.utils.data as torch_data

from coach.config import CfgNode, configurable
from coach.utils.comm import get_world_size
from coach.utils.env import seed_all_rng

from .catalog import DatasetCatalogSingleton

__all__ = [
    "build_coach_train_loader",
    "build_coach_test_loader"
]

def get_dataset(names: list[str], params: list[list[Any]]) -> torch_data.Dataset:
    """
    Get the dataset dicts from the dataset catalog.

    Args:
        names (list[str]): The names of the datasets to get.
        params (list[list[Any]]): The parameters to pass to the dataset constructor,
            in the same order as the names.

    Returns:
        torch_data.Dataset: The single dataset or the concatenated dataset.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names) > 0, "Must provide at least one dataset name."

    available_datasets = DatasetCatalogSingleton.keys()
    names_set = set(names)
    if not names_set.issubset(available_datasets):
        logger = logging.getLogger(__name__)
        logger.warning("Dataset {} is not available in the dataset catalog.".format(
            names_set - available_datasets
        ))

    dataset = [DatasetCatalogSingleton.get(name, *param) for name, param in zip(names, params)]

    if len(dataset) > 1:
        return torch_data.ConcatDataset(dataset)
    else:
        return dataset[0]

def _train_loader_from_config(cfg: CfgNode) -> dict[str, Any]:
    logger = logging.getLogger(__name__)

    dataset = get_dataset(cfg.DATASETS.TRAIN.NAMES, cfg.DATASETS.TRAIN.PARAMS)

    if isinstance(dataset, torch_data.IterableDataset):
        logger.info("Using iterable dataset, sampler will be ignored.")
        sampler = None
    else:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger.info("Using dataset with sampler {}.".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        else:
            raise KeyError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "image_batch_size": cfg.DATALOADER.IMAGE_BATCH_SIZE,
        "ray_batch_size": cfg.DATALOADER.RAY_BATCH_SIZE,
        "num_workers": cfg.DATALOADER.NUM_WORKERS
    }

@configurable(from_config=_train_loader_from_config)
def build_coach_train_loader(
    dataset,
    *,
    sampler: torch_data.Sampler | None = None,
    image_batch_size: int,
    ray_batch_size: int,
    num_workers: int = 0,
    **kwargs
) -> torch_data.DataLoader:
    """
    Build the train loader for Coach.

    Args:
    """
    if isinstance(dataset, torch_data.IterableDataset):
        assert sampler is None, "sampler must be None for IterableDataset."

    world_size = get_world_size()
    total_image_batch_size = ray_batch_size * world_size
    total_ray_batch_size = image_batch_size * world_size
    logger = logging.getLogger(__name__)
    logger.info("Building train loader with image batch size {} and ray batch size {}.".format(
        total_image_batch_size, total_ray_batch_size
    ))

    return torch_data.DataLoader(
        dataset,
        batch_size=total_image_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_reset_seed,
        **kwargs
    )

def _test_loader_from_config(cfg: CfgNode) -> dict[str, Any]:
    return {}

@configurable(from_config=_test_loader_from_config)
def build_coach_test_loader(
    dataset,
    *,
    sampler: torch_data.Sampler | None = None,
    image_batch_size: int = 1,
    num_workers: int = 0
) -> torch_data.DataLoader:
    return torch_data.DataLoader(
        dataset,
        batch_size=image_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def collate_fn(batch: Any) -> Any:
    """
    Collate function for the train loader.
    """
    return batch

def worker_init_reset_seed(worker_id: int):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)
