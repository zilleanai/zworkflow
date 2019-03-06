# source: https://medium.com/@alexrachnog/financial-forecasting-with-probabilistic-programming-and-pyro-db68ab1a1dba
# source: http://pyro.ai/examples/bayesian_regression.html
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.contrib.gp as gp
from pyro.distributions import Normal, Uniform, Delta
from pyro.contrib.autoguide import AutoDiagonalNormal

from mlworkflow.model import ModelBase


class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.linear(x) + (self.factor * x[:, 0] * x[:, 1]).unsqueeze(1)


class model(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.initModel()

    def initModel(self):
        self.regression_model = RegressionModel(len(self.features))
        self.kernel = gp.kernels.RBF(input_dim=len(self.features), variance=torch.tensor(5.),
                                     lengthscale=torch.tensor(10.))
        self.guide = AutoDiagonalNormal(self.model)

    def model(self, x_data, y_data):
        # weight and bias priors
        w_prior = Normal(torch.zeros(1, 2), torch.ones(1, 2)).to_event(1)
        b_prior = Normal(torch.tensor([[8.]]),
                         torch.tensor([[1000.]])).to_event(1)
        f_prior = Normal(0., 1.)
        priors = {'linear.weight': w_prior,
                  'linear.bias': b_prior, 'factor': f_prior}
        scale = pyro.sample("sigma", Uniform(0., 10.))
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module(
            "module", self.regression_model, priors)
        # sample a nn (which also samples w and b)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(x_data)):
            # run the nn forward on data
            prediction_mean = lifted_reg_model(x_data).squeeze(-1)
            # condition on the observed data
            pyro.sample("obs",
                        Normal(prediction_mean, scale),
                        obs=y_data)
            return prediction_mean

    def net(self):
        return self.kernel

    def __str__(self):
        return str(self.features) + ', ' + str(self.labels)

    def save(self):
        torch.save(self.kernel.state_dict(), self.config['model']['savepath'])
        if self.config['general']['verbose']:
            print('saved model: ', self.config['model']['savepath'])

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            self.kernel.load_state_dict(
                torch.load(self.config['model']['savepath']))
            if self.config['general']['verbose']:
                print('loaded model: ', self.config['model']['savepath'])
