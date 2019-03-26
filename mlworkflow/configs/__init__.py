from .default import default
from .bayesian_optimization import bayesian_optimization
from .csv import csv
from .probabilistic_programming import probabilistic_programming
from .segmentation import segmentation

configs = {
    'default': default,
    'bayesian_optimization': bayesian_optimization,
    'csv': csv,
    'probabilistic_programming': probabilistic_programming,
    'segmentation': segmentation
}


def get_config(domain='default'):
    return configs.get(domain) or default
