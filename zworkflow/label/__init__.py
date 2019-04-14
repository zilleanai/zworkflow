import os
import importlib.util
from importlib import import_module

from .labelbase import LabelBase

label_classes = {
}


def available():
    return list(label_classes.keys())


def get_label(config):
    class_name = config['label']['label_class']
    label_file = config['label']['label_file']
    if class_name in available():
        return label_classes[class_name](config)
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(label_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
