import os
import importlib.util
from importlib import import_module

from .trainbase import TrainBase


def get_train(config):
    class_name = config['train']['train_class']
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(config['train']['train_file']))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
