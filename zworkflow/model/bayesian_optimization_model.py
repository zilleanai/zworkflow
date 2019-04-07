import os
import json
import yaml
import inspect

from .modelbase import ModelBase


class BayesianOptimizationModel(ModelBase):

    data = None
    target = None

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.res = []
        self.max = {}
        self.max['params'] = self.default_params()
        self.pbounds = self.default_param_bounds()

    def default_params(self):
        params = {}
        params_keys = inspect.getfullargspec(self.f)[0][2:]
        params_values = inspect.getfullargspec(self.f)[3]

        for i, param in enumerate(params_keys):
            params[param] = params_values[i]
        for param in self.config['model']['params']:
            params[param] = self.config['model']['params'][param]
        return params

    def default_param_bounds(self):
        pbounds = {}
        for param in self.max['params']:
            pbounds[param] = (-1, 1)
        for param in self.config['model']['param_bounds']:
            pbounds[param] = tuple(self.config['model']['param_bounds'][param])
        return pbounds

    def f(self, data):
        raise NotImplementedError

    def optimizable(self):
        raise NotImplementedError

    def __str__(self):
        return 'bayesian optimization: ' + str(self.features) + ', ' + str(self.labels)

    def save(self):
        with open(self.config['model']['savepath'], 'w') as outfile:
            json.dump(self.max, outfile, indent=4)

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            with open(self.config['model']['savepath']) as file:
                self.max = json.load(file)
                self.res = [self.max]

    def load_logs(self, optimizer):
        logs = self.res
        for log in logs:
            optimizer.register(
                params=log["params"],
                target=log["target"],
            )
