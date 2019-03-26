import os
import importlib.util
from importlib import import_module

from .predictbase import PredictBase
from .bayesian_optimization_predict import BayesianOptimizationPredict
from .csv_predict import CSVPredict
from .probabilistic_programming_predict import ProbabilisticProgrammingPredict
from .segmentation_predict import SegmentationPredict


predict_classes = {
    'bayesian_optimization_predict': BayesianOptimizationPredict,
    'csv_predict': CSVPredict,
    'probabilistic_programming_predict': ProbabilisticProgrammingPredict,
    'segmentation_predict': SegmentationPredict
}


def available():
    return list(predict_classes.keys())


def get_predict(config, preprocessing):
    class_name = config['predict']['predict_class']
    predict_file = config['predict']['predict_file']
    if class_name in available():
        return predict_classes[class_name](config, preprocessing)
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(predict_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config, preprocessing)
