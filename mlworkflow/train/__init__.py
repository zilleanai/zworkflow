import os
import importlib.util
from importlib import import_module

from .trainbase import TrainBase
from .bayesian_optimization_train import BayesianOptimizationTrain
from .probabilistic_programming_train import ProbabilisticProgrammingTrain

train_classes = {
    'bayesian_optimization_trainer': BayesianOptimizationTrain,
    'probabilistic_programming_train': ProbabilisticProgrammingTrain
}


def available():
    return list(train_classes.keys())


def get_train(config):
    class_name = config['train']['train_class']
    train_file = config['train']['train_file']
    if class_name in available():
        return train_classes[class_name](config)
    spec = importlib.util.spec_from_file_location(
        class_name, os.path.join(train_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_attr = getattr(module, class_name)
    return class_attr(config)
