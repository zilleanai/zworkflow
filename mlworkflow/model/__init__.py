import os
import importlib.util
from importlib import import_module

from .modelbase import ModelBase
from .bayesian_optimization_model import BayesianOptimizationModel
from .probabilistic_programming_model import ProbabilisticProgrammingModel
from .segmentation_model import SegmentationModel

model_classes = {
    'bayesian_optimization_model': BayesianOptimizationModel,
    'probabilistic_programming_model': ProbabilisticProgrammingModel,
    'segmentation_model': SegmentationModel
}


def available():
    return list(model_classes.keys())


def get_model(config):
    class_name = config['model']['model_class']
    model_file = config['model']['model_file']
    if class_name in available():
        return model_classes[class_name](config)
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(model_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
