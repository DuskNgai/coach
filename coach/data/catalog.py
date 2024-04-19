from collections import UserDict
import inspect
from typing import Any

import torch.utils.data as torch_data

__all__ = [
    "DatasetCatalogSingleton"
]


class DatasetCatalog(UserDict):
    """
    A global dictionary that stores information about the datasets and how to obtain them.

    It contains a mapping from strings (which are names that identify a dataset, e.g. "?")
    to a class

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    def register(self, name: str, cls) -> None:
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "?".
        """
        assert inspect.isclass(cls), "You must register a function with `DatasetCatalog.register`!"
        assert name not in self, "Dataset '{}' is already registered!".format(name)
        self[name] = cls

    def get(self, name: str, *args: Any, **kwargs: Any) -> torch_data.Dataset:
        """
        Call the registered function and return its results.

        Args:
            name (str): the name that identifies a dataset, e.g. "?".
            args (Any): the parameters to pass to the dataset constructor.
            kwargs (Any): the parameters to pass to the dataset constructor.

        Returns:
            torch_data.Dataset: the dataset loaded from the given path.
        """
        try:
            cls = self[name]
        except KeyError as e:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name, ", ".join(list(self.keys()))
                )
            ) from e
        return cls(*args, **kwargs)

    def list(self) -> list[str]:
        """
        List all registered datasets.

        Returns:
            list[str]
        """
        return list(self.keys())

    def pop(self, name: str) -> None:
        """
        Remove the dataset with `name` from the registry and return the
        corresponding function.

        Args:
            name (str): the name that identifies a dataset, e.g. "?".
        """
        self.pop(name)

    def __str__(self):
        return "DatasetCatalog(registered datasets: {})".format(", ".join(self.keys()))

    __repr__ = __str__

DatasetCatalogSingleton = DatasetCatalog()
DatasetCatalogSingleton.__doc__ = (
    DatasetCatalog.__doc__ + "\n" + DatasetCatalog.register.__doc__ + "\n" + DatasetCatalog.get.__doc__
)
