import os
import importlib.util
from importlib import import_module

from .evaluatebase import EvaluateBase

evaluate_classes = {
}


def available():
    return list(evaluate_classes.keys())


def get_evaluate(config):
    class_name = config['evaluate']['evaluate_class']
    predict_file = config['evaluate']['evaluate_file']
    if class_name in available():
        return evaluate_classes[class_name](config)
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(predict_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
