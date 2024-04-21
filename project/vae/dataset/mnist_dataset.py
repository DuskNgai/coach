import torch
from torchvision import datasets, transforms

class MNISTDataset(datasets.MNIST):
    def __init__(self,
        root: str,
        is_train: bool,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            root=root,
            train=is_train,
            transform=transforms.ToTensor(),
            *args,
            **kwargs
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        image, target = super().__getitem__(index)
        return image

    @property
    def collate_fn(self):
        return None
