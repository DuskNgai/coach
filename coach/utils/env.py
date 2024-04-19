import datetime
import importlib
import importlib.util
import logging
import os
import random
import sys

import numpy as np
import torch

__all__ = ["seed_all_rng"]


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
"""
PyTorch version as a tuple of 2 ints. Useful for comparison.
"""

def seed_all_rng(seed: int = None) -> None:
    """
    Seed all the random number generators in random, numpy and torch.

    Args:
        `seed` (int): The seed to use. If None, a random seed will be generated.
    """
    if seed is None:
        seed = os.getpid() + int(datetime.datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _configure_libraries() -> None:
    """
    Configure some libraries for better performance.
    """
    # Disable OpenCV in case it leads to negative performance impact.
    disable_cv2 = os.environ.get("COACH_DISABLE_OPENCV", False)
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable OpenCL in OpenCV since its interaction with CUDA which leads to negative performance impact.
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2
            # This environment variable is only available after OpenCV 3.4.0.
            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ModuleNotFoundError:
            # OpenCV is not installed.
            pass

def _import_file(module_name: str, file_name: str, make_importable: bool) -> type(sys):
    """
    Import a Python file as a module from a file path.

    Args:
        `module_name` (str): The name of the module that can be imported.
        `file_name` (str): The path to the Python file.
        `make_importable` (bool): Whether it can be imported by other modules.

    Returns:
        (importlib.ModuleType): The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module

def setup_env_module(env_module_path: str) -> None:
    """
    Load a custom environment module from a Python file or a module.
    And then run its `setup_environment()`.

    Args:
        `env_module_path` (str): The path to the custom environment module.
    """
    if env_module_path.endswith(".py"):
        env_module_path = _import_file("detectron2.utils.env.custom_module", env_module_path)
    else:
        module = importlib.import_module(env_module_path)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the required callable attribute `setup_environment`."
    ).format(env_module_path)
    module.setup_environment()

_ENV_SETUP_DONE = False

def setup_environment() -> None:
    """
    Setup the environment. If the environment variable `COACH_ENV_MODULE` is set,
    a Python file or a module will be loaded. Its `setup_environment()` will be called.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    env_module_path = os.environ.get("COACH_ENV_MODULE", None)

    if env_module_path is not None:
        setup_env_module(env_module_path)
