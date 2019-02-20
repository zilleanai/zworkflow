import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlworkflow.model import ModelBase


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size*5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size*5, input_size*3)
        self.fc3 = nn.Linear(input_size*3, input_size)
        self.fc4 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class model(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.initNet()

    def initNet(self):
        self.__net = NeuralNet(len(self.features), len(self.labels))


    def net(self):
        return self.__net

    def __str__(self):
        return str(self.features) + ', ' + str(self.labels)

    def save(self):
        torch.save(self.__net.state_dict(), self.config['model']['savepath'])
        print('saved model: ', self.config['model']['savepath'])

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            self.__net.load_state_dict(torch.load(self.config['model']['savepath']))
            print('loaded model: ', self.config['model']['savepath'])