from coach.data import DatasetCatalogSingleton

from .blender_dataset import BlenderDataset

def register_dataset():
    DatasetCatalogSingleton.register("BlenderDataset", BlenderDataset)

if __name__.endswith(".register"):
    register_dataset()
