import os
import json
import yaml
import inspect

from mlworkflow.model import BayesianOptimizationModel

class model(BayesianOptimizationModel):


    def __init__(self, config):
        super().__init__(config)

    def f(self, data, param1=1, param2=2, param3=5.1):
        return data[:, 0] * param1 - data[:, 1] * param2 + data[:, 1] * param3

    def optimizable(self, param1, param2, param3):
        f = self.f(self.data, param1, param2, param3)
        diff = (f - self.target[:, 0])**2
        return 1-diff.mean()
