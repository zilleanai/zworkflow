import os
import importlib.util
from importlib import import_module

from .labelbase import LabelBase


def get_label(config):
    class_name = 'label'
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join('label.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
