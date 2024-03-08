import json
import math
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils import data
import tqdm

class BlenderDataset(data.Dataset):
    def __init__(self, root_dir: str, split: str = "train"):
        super(BlenderDataset, self).__init__()

        #* Read json file.
        transforms_path = Path(root_dir).joinpath("transforms_{}.json".format(split))
        with open(transforms_path, mode='r') as f:
            transforms = json.load(f)

        #* load images and poses
        images, poses = [], []
        for frame in tqdm.tqdm(transforms["frames"], desc="Loading {} images and poses".format(split)):
            #* load poses
            pose = torch.Tensor(frame["transform_matrix"])
            poses.append(pose)

            #* load images
            image_path = Path(root_dir).joinpath(frame["file_path"])
            if not image_path.suffix:
                image_path = image_path.with_suffix(".png")

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
        H, W, _ = images[0].shape
        self.images = torch.stack(images)           # [N, H, W, C]
        self.poses = torch.stack(poses)             # [N, 4, 4]
        self.extrinsics = torch.inverse(self.poses) # [N, 4, 4]
        self.n_views = self.images.shape[0]

        flx = 0.5 * W / math.tan(0.5 * transforms["camera_angle_x"])
        fly = 0.5 * H / math.tan(0.5 * transforms["camera_angle_y"] if "camera_angle_y" in transforms else transforms["camera_angle_x"])
        cx = W / 2
        cy = H / 2
        self.intrinsic = torch.Tensor([[flx, 0.0, cx], [0.0, fly, cy], [0.0, 0.0, 1.0]])

    def __len__(self) -> int:
        return self.n_views

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "pose": self.poses[index],
            "intrinsic": self.intrinsic,
        }
