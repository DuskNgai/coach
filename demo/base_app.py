from argparse import Namespace
from pathlib import Path
import random

import torch
import numpy as np

from utils import configurations

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    Args:
        seed (int): The random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class BaseApp(object):
    def __init__(self, args: Namespace) -> None:
        print("Creating output directory.")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        #* Saving a copy of the configuration file to the output directory.
        self.config = configurations.merge_config_with_args(
            configurations.load_config(args.config_path), args
        )
        configurations.save_config(self.config, self.output_dir.joinpath("config.toml"))

        #* Set device, data type, anomaly detection and random seed.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float32)
        torch.autograd.set_detect_anomaly(True)
        set_random_seed(19268017)

    def run(self):
        raise NotImplementedError()
