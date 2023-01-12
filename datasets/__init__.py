"""This package includes all the modules related to data loading and preprocessing.

    To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
"""
import importlib
from datasets.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In {0}.py, there should be a subclass of BaseDataset with class name that matches {1} in lowercase.".format(
                dataset_filename, target_dataset_name
            )
        )

    return dataset


def get_dataset(config):
    """Return initialized dataset class from configuration (loaded from the config file).

    Example:
        from datasets import get_dataset
        dataset = get_dataset(configuration)
    """
    ds_class = find_dataset_using_name(config["name"])
    return ds_class(config)
