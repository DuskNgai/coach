from torch.utils import data

from utils import logger
from ray_dataset import SyntheticDataset

logger = logger.Logger("dataset/build")

def build_train_loader(config: dict) -> data.DataLoader:

    batch_size = config.getint("dataset", "batch_size")
    dataset_name = config.get("dataset", "dataset_name")
    if dataset_name == "synthetic":
        dataset = SyntheticDataset(config)
    elif dataset_name == "real":
        raise NotImplementedError("Real dataset is not implemented yet.")
    else:
        raise ValueError(
            "Wrong dataset name: got `{}`. Valid names are `synthetic` and `real`.".format(dataset_name)
        )

    n_workers = config.getint("dataset", "n_workers")
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, 
    )

    return data_loader
