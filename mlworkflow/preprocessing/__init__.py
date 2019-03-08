import os
import importlib.util
from importlib import import_module

from .preprocessingbase import PreprocessingBase


def get_preprocessing(config):
    class_name = 'preprocessing'
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join('preprocessing.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
