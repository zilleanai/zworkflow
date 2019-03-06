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
        self.conv1 = nn.Conv1d(input_size, 32, 1)
        self.conv2 = nn.Conv1d(32, 96, 3, padding=3)
        self.pool1 = nn.MaxPool1d(7, padding=3)
        self.conv3 = nn.Conv1d(96, 96, 3, padding=3)
        self.pool2 = nn.MaxPool1d(7, padding=3)
        self.conv4 = nn.Conv1d(96, 32, 1)
        self.conv5 = nn.Conv1d(32, input_size, 1)
        self.conv6 = nn.Conv1d(input_size, 1, 1)
        self.fc2 = nn.Linear(input_size*5, input_size*3)
        self.fc3 = nn.Linear(input_size*3, input_size)
        self.fc4 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x[:,:, None]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool2(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        #out = self.fc4(out)
        out = torch.squeeze(out)
        return out

    # source: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class model(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.initNet()

    def initNet(self):
        self.__net = NeuralNet(len(self.features), len(self.labels))
        self.__net.apply(self.__net.init_weights)


    def net(self):
        return self.__net

    def __str__(self):
        return str(self.features) + ', ' + str(self.labels)

    def save(self):
        torch.save(self.__net.state_dict(), self.config['model']['savepath'])
        if self.config['general']['verbose']:
            print('saved model: ', self.config['model']['savepath'])

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            self.__net.load_state_dict(torch.load(self.config['model']['savepath']))
            if self.config['general']['verbose']:
                print('loaded model: ', self.config['model']['savepath'])