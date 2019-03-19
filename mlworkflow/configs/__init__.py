from .default import default
from .bayesian_optimization import bayesian_optimization
from .probabilistic_programming import probabilistic_programming

configs = {
    'default': default,
    'bayesian_optimization': bayesian_optimization,
    'probabilistic_programming': probabilistic_programming
}


def get_config(domain='default'):
    return configs.get(domain) or default
