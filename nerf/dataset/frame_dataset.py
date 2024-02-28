import json
import math
from pathlib import Path

import torch
import tqdm
from torch.utils import data

from io_utils import ImageIO

class SyntheticImageDataset(data.Dataset):
    def __init__(self, config: dict):
        super(SyntheticImageDataset, self).__init__()

        dataset_path = config["dataset"]["dataset_path"]

        #* Read json file.
        json_file_name = Path(dataset_path).joinpath("transforms.json")
        with open(json_file_name, mode='r') as f:
            json_file = json.load(f)

        self.angles = json_file["camera_angle_x"]
        self.n_camera = len(json_file["frames"])

        #* load images and poses
        images = []
        poses = []
        scaling_factor = config.getint("dataset", "scaling_factor")
        for i in tqdm.tqdm(range(self.n_camera)):
            frame = json_file["frames"][i]

            #* load poses
            poses = torch.Tensor(frame["transform_matrix"])
            poses.append(poses)

            #* load images
            image_path = Path(dataset_path).join(frame["file_path"])
            image = ImageIO.input(image_path)
            if scaling_factor != 1:
                image = image.resize((image.width // scaling_factor, image.height // scaling_factor))

            # blend alpha channel to RGB
            if image.shape[2] == 4:
                image = image[0:3] * image[3] + (1.0 - image[3])

            images.append(image)

        #* camera and image settings
        C, H, W = image[0].shape
        # self.images.size() = (N, C, H, W)
        self.images = torch.stack(images)
        # self.poses.size() = (N, 4, 4)
        self.poses = torch.stack(poses)
        # self.extrinsics.size() = (N, 4, 4)
        self.extrinsics = torch.inverse(self.poses)

        flx = 0.5 * W / math.tan(0.5 * json_file["camera_angle_x"])
        fly = 0.5 * H / math.tan(0.5 * json_file["camera_angle_y"])
        cx = W / 2
        cy = H / 2
        self.intrinsic = torch.Tensor([[flx, 0.0, cx], [0.0, fly, cy], [0.0, 0.0, 1.0]])

        #* Get `near_fars` for each image.
        self.translation = self.poses[:, 0:3, 3:4]
        T = self.poses[:, 0:4, 3:4]


    def __len__(self) -> int:
        return self.n_camera

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.poses[index], self.intrinsic, self.images[index], self.mask, self.near_far
