import os
import json
import yaml

from mlworkflow.model import ModelBase

from bayes_opt import BayesianOptimization


class model(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.res = []
        self.max = {"params": {
                    "param1": 1,
                    "param2": 1
                    }}

    def f(self, data, param1, param2):
        return data[:, 0] * param1 - data[:, 1] * param2

    def train(self, data, target):

        def optimizable(param1, param2):
            f = self.f(data, param1, param2)
            diff = (f - target[:, 0])**2
            return 1-diff.mean()

        pbounds = {'param1': (-10, 10), 'param2': (-100, 100)}

        optimizer = BayesianOptimization(
            f=optimizable,
            pbounds=pbounds,
            verbose=2,
            random_state=20,
        )
        self.load_logs(optimizer, logs=self.res)
        optimizer.maximize(
            init_points=10,
            n_iter=self.config['train']['epochs']
        )
        self.res = optimizer.res
        self.max = optimizer.max

    def predict(self, data):
        return self.f(data, self.max['params']['param1'], self.max['params']['param2'])

    def __str__(self):
        return str(self.features) + ', ' + str(self.labels)

    def save(self):
        with open(self.config['model']['savepath'], 'w') as outfile:
            json.dump(self.max, outfile, indent=4)

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            with open(self.config['model']['savepath']) as file:
                self.max = json.load(file)
                self.res = [self.max]

    def load_logs(self, optimizer, logs):
        for log in logs:
            optimizer.register(
                params=log["params"],
                target=log["target"],
            )
