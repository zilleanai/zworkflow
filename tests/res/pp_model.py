# source: https://medium.com/@alexrachnog/financial-forecasting-with-probabilistic-programming-and-pyro-db68ab1a1dba
# source: http://pyro.ai/examples/bayesian_regression.html
# source: https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/ss_vae_M2.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.contrib.gp as gp
from pyro.distributions import Normal, Uniform, Delta, Bernoulli
from pyro.contrib.autoguide import AutoDiagonalNormal
import pyro.distributions as dist

from mlworkflow.model import ModelBase


class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        # p = number of features
        super(DNN, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, input_size*2)
        self.fc2 = nn.Linear(input_size*2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


class PPModel(nn.Module):
    def __init__(self, input_size, output_size=1, z_dim=16):
        # p = number of features
        super(PPModel, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.encoder_y = DNN(self.input_size, self.output_size)
        self.encoder_z = DNN(
            self.input_size, self.z_dim)
        self.decoder = DNN(self.z_dim + self.output_size,
                           self.input_size)

    def model(self, xs, ys=None):
        pyro.module('pp_model', self)
        batch_size = xs.size(0)
        with pyro.plate('data'):
            prior_loc = xs.new_zeros([batch_size, self.z_dim])
            prior_scale = xs.new_ones([batch_size, self.z_dim])
            zs = pyro.sample("z", Normal(prior_loc, prior_scale).to_event(1))
            alpha_prior = xs.new_ones([batch_size, self.output_size]) / (1.0*self.output_size)
            ys = pyro.sample("y", Uniform(alpha_prior), obs=ys)
            loc = self.decoder.forward([zs, ys])
            pyro.sample('x', Bernoulli(loc).to_event(1), obs=xs)
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", Uniform(alpha))
            loc, scale = self.encoder_z.forward(xs)
            pyro.sample("z", Normal(loc, scale).to_event(1))

    def predict(self, xs):
        alpha = self.encoder_y.forward(xs)
        ys = xs.new_zeros(alpha.size())
        return ys



class model(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.initModel()

    def initModel(self):
        self.__model = PPModel(len(self.features), len(self.labels))

    def model(self):
        return self.__model

    def __str__(self):
        return str(self.features) + ', ' + str(self.labels)

    def save(self):
        torch.save(self.__model.state_dict(), self.config['model']['savepath'])
        if self.config['general']['verbose']:
            print('saved model: ', self.config['model']['savepath'])

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            self.__model.load_state_dict(
                torch.load(self.config['model']['savepath']))
            if self.config['general']['verbose']:
                print('loaded model: ', self.config['model']['savepath'])
