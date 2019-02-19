import os
import importlib.util
from importlib import import_module

from .datasetbase import DataSetBase


def get_dataset(config):
    class_name = 'dataset'
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join('dataset.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
