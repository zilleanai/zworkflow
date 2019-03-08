import os
import importlib.util
from importlib import import_module

from .predictbase import PredictBase


def get_predict(config, preprocessing):
    class_name = config['predict']['predict_class']
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(config['predict']['predict_file']))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config, preprocessing)
