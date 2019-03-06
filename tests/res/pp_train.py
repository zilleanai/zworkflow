# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pyro
import pyro.contrib.gp as gp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.optim import Adam
import pyro.optim as optim

from mlworkflow.train import TrainBase


class train(TrainBase):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config['train']['device'])
        self.criterion = nn.MSELoss().to(self.device)

    def train(self, dataset, model, logger=print):
        kernel = model.net()
        kernel.to(self.device)
        if self.config['train']['load_model']:
            model.load()
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        
        epochs = self.config['train']['epochs']
        total_step = len(dataset) // self.config['train']['batch_size']
        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'],
                                shuffle=True, num_workers=4)

        optim = Adam({"lr": 0.03})
        svi = SVI(model.model, model.guide, optim, loss=Trace_ELBO(), num_samples=1000)

        for epoch in range(epochs):
            logger('epoch: ', epoch)
            for i, (X, y) in tqdm(enumerate(dataloader)):
                X = X.view(-1, len(self.config['dataset']['features']))
                X = X.to(self.device)
                y = y.to(self.device)

                loss = svi.step(X, y)
                if (i+1) % 2 == 0:
                    logger('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                               .format(epoch+1, epochs, i+1, total_step, loss/total_step))
            if epoch % self.config['train']['save_every_epoch'] == 0:
                model.save()

    def __str__(self):
        return 'pp trainer'
