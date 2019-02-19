# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mlworkflow.train import TrainBase


class train(TrainBase):

    def __init__(self, config):
        super().__init__(config)
        self.criterion = nn.MSELoss()

    def train(self, dataset, model):
        net = model.net()
        criterion = torch.nn.MSELoss(size_average = False)
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.config['train']['learn_rate'])
        epochs = self.config['train']['epochs']
        total_step = len(dataset)
        for epoch in range(epochs-1):
            print('epoch: ', epoch)
            for i, (X, y) in tqdm(enumerate(dataset)):
                X = Variable(torch.from_numpy(X)).view(-1, len(self.config['dataset']['features']))
                y = Variable(torch.from_numpy(y))
                outputs = net(X)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 1000 == 0:
                    tqdm.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, epochs, i+1, total_step, loss.item()))

    def __str__(self):
        return 'my trainer'
