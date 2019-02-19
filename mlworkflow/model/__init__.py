import os
import importlib.util
from importlib import import_module

from .modelbase import ModelBase


def get_model(config):
    class_name = 'model'
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join('model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
