import os
import importlib.util
from importlib import import_module

from .datasetbase import DataSetBase
from .csv_dataset import CSVDataset
from .segmentation_dataset import SegmentationDataset

dataset_classes = {
    'csv_dataset': CSVDataset,
    'segmentation_dataset': SegmentationDataset
}


def available():
    return list(dataset_classes.keys())


def get_dataset(config, preprocessing=None):
    class_name = config['dataset']['dataset_class']
    dataset_file = config['dataset']['dataset_file']
    if class_name in available():
        return dataset_classes[class_name](config, preprocessing)
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(dataset_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config, preprocessing)
