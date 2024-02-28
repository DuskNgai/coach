from typing import Tuple

import torch
import torch.nn.functional as F

def sample_ray_synthetic(pose: torch.Tensor, intrinsic: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain rays and pixel colors from camera pose, intrinsic and image."""
    C, H, W = image.size()
    # x, y.size() = (H, W)
    x, y = torch.meshgrid(
        torch.linspace(0.0, W - 1, W, dtype=torch.float32),
        torch.linspace(0.0, H - 1, H, dtype=torch.float32),
        indexing="xy"
    )

    transform = pose[0:3, 0:3] @ torch.inverse(intrinsic)
    direction = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    direction = torch.einsum("ij, hwj->hwi", transform, direction)
    # ray_d, ray_o.size() = (H, W, 3)
    ray_d = F.normalize(direction, dim=-1)
    ray_o = pose[0:3, 3].expand(ray_d.shape)

    # rays.size() = (H * W, 6)
    rays = torch.cat([ray_o, ray_d], dim=0).reshape(-1, 6)
    # colors.size() = (H * W, 3)
    colors = image.reshape(3, -1).transpose(0, 1)

    return rays, colors
