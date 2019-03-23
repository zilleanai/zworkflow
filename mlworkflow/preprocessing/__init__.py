import os
import importlib.util
from importlib import import_module

from .preprocessingbase import PreprocessingBase
from .segmentation_preprocessing import SegmentationPreprocessing

preprocessing_classes = {
    'segmentation_preprocessing': SegmentationPreprocessing
}


def available():
    return list(preprocessing_classes.keys())


def get_preprocessing(config):
    class_name = config['preprocessing']['preprocessing_class']
    dataset_file = config['preprocessing']['preprocessing_file']
    if class_name in available():
        return preprocessing_classes[class_name](config)
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(dataset_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)

