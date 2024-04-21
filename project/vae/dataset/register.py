from coach.data import DatasetCatalogSingleton

from .mnist_dataset import MNISTDataset

def register_dataset():
    DatasetCatalogSingleton.register("MNISTDataset", MNISTDataset)

if __name__.endswith(".register"):
    register_dataset()
