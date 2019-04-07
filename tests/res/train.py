# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from zworkflow.train import TrainBase


class train(TrainBase):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config['train']['device'])
        self.criterion = nn.MSELoss().to(self.device)

    def train(self, dataset, model, logger=print):
        net = model.net()
        net.to(self.device)
        if self.config['train']['load_model']:
            model.load()
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.Adagrad(
            net.parameters(), lr=self.config['train']['learn_rate'], weight_decay=1e-5)
        epochs = self.config['train']['epochs']
        total_step = len(dataset) // self.config['train']['batch_size']
        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'],
                                shuffle=True, num_workers=4)
        for epoch in range(epochs):
            logger('epoch: ', epoch)
            for i, (X, y) in tqdm(enumerate(dataloader)):
                X = X.view(-1, len(self.config['dataset']['features']))
                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                outputs = net(X)
                loss = criterion(outputs, y)
                
                loss.backward()
                optimizer.step()
                if (i+1) % 2 == 0:
                    logger('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                               .format(epoch+1, epochs, i+1, total_step, loss.item()))
            if epoch % self.config['train']['save_every_epoch'] == 0:
                model.save()
        model.save()

    def __str__(self):
        return 'my trainer'
