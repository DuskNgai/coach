import json
import math
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torchvision.transforms
from torch.utils import data
import tqdm

class BlenderDataset(data.Dataset):
    def __init__(self,
        config: dict
    ):
        super(BlenderDataset, self).__init__()

        dataset_path = config["dataset"]["dataset_path"]

        #* Read json file.
        transforms_path = Path(dataset_path).joinpath("transforms.json")
        with open(transforms_path, mode='r') as f:
            transforms = json.load(f)

        self.angles = transforms["camera_angle_x"]
        self.n_camera = len(transforms["frames"])

        #* load images and poses
        images, poses = [], []
        for frame in tqdm.tqdm(transforms["frames"]):
            #* load poses
            poses = torch.Tensor(frame["transform_matrix"])
            poses.append(poses)

            #* load images
            image_path = Path(dataset_path).join(frame["file_path"])
            with Image.open(image_path) as img:
                image = torch.from_numpy(np.array(img) / 255.0) # [H, W, C]

            # blend alpha channel to RGB
            if image.ndim == 2:
                image = image.unsqueeze(-1).expand(-1, -1, 3)
            if image.shape[2] == 4:
                rgb = image[..., 0:3]
                alpha = image[..., 3:4]
                image = alpha * rgb + (1 - alpha) * 1.0

            images.append(image)

        #* camera and image settings
        H, W, C = image[0].shape
        self.images = torch.stack(images)           # [N, H, W, C]
        self.poses = torch.stack(poses)             # [N, 4, 4]
        self.extrinsics = torch.inverse(self.poses) # [N, 4, 4]

        flx = 0.5 * W / math.tan(0.5 * transforms["camera_angle_x"])
        fly = 0.5 * H / math.tan(0.5 * transforms["camera_angle_y"])
        cx = W / 2
        cy = H / 2
        self.intrinsic = torch.Tensor([[flx, 0.0, cx], [0.0, fly, cy], [0.0, 0.0, 1.0]])

        #* Get `near_fars` for each image.
        self.translation = self.poses[:, 0:3, 3:4]


    def __len__(self) -> int:
        return self.n_camera

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.poses[index], self.intrinsic, self.images[index], self.mask, self.near_far
