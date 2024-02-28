from pathlib import Path
from typing import Tuple

import torch
from torch.utils import data
import tqdm

from .frame_dataset import SyntheticImageDataset
from .ray_sampling import sample_ray_synthetic
from utils import logger

logger = logger.Logger("dataset/ray_dataset")

class RayDataset(data.Dataset):
    def __init__(self, config: dict) -> None:
        super(RayDataset, self).__init__()

        #* IO settings.
        self.output_dir = config["output"]["output_dir"]
        self.dataset_path = config["dataset"]["dataset_path"]

        if not Path(self.dataset_path).exists():
            raise FileExistsError("{} does not exist".format(self.dataset_path))

        #* Dataset settings.
        self.image_dataset = SyntheticImageDataset(config)
        self.angles = self.image_dataset.angles
        self.camera_num = self.image_dataset.camera_num

        #* Generating rays.
        logger.info("Generating rays...")
        ray_list = []
        color_list = []
        near_far_list = []

        process_bar = tqdm.tqdm(range(len(self.image_dataset)), desc="Generating Image No.00 rays")
        for i in process_bar:
            # near_far.size() = (???)
            poses, intrinsic, image, mask, near_far = self.image_dataset[i]

            # rays.size() = (H * W, 6)
            # colors.size() = (H * W, 3)
            rays, colors = sample_ray_synthetic(poses, intrinsic, image)
            ray_list.append(rays)
            color_list.append(colors)
            near_far_list.append(near_far)
            
            process_bar.set_description("Generating Image No.{:02d} rays".format(i + 1))

        # self.rays.size() = (N * H * W, 6)
        self.rays = torch.cat(ray_list, dim=0)
        # self.colors.size() = (N * H * W, 6)
        self.colors = torch.cat(color_list, dim=0)
        # self.near_fars.size() = (???)
        self.near_fars = torch.cat(near_far_list, dim=0)

    def __len__(self) -> int:
        return self.rays.size(1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.rays[:, index],
            self.colors[:, index],
            self.near_fars[:, index]
        )
