# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from .trainbase import TrainBase

# source: https://github.com/usuyama/pytorch-unet/blob/master/loss.py
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class SegmentationTrain(TrainBase):


    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config['train']['device'])
        self.criterion = dice_loss or nn.CrossEntropyLoss().to(self.device)
        self.writer = SummaryWriter(self.config['train'].get(
            'tensorboard')) if self.config['train'].get('tensorboard') else None

    def train(self, dataset, model, logger=print):
        net = model.net()
        net.to(self.device)
        if self.config['train']['load_model']:
            model.load()
        optimizer = torch.optim.Adagrad(
            net.parameters(), lr=self.config['train']['learn_rate'], weight_decay=1e-5)
        epochs = self.config['train']['epochs']
        N = len(dataset)
        total_step = N // self.config['train']['batch_size']
        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'],
                                shuffle=True, num_workers=4)
        n_iter = 0
        for epoch in range(epochs):
            logger('epoch: ', epoch)
            epoch_loss = 0.0
            t = tqdm(enumerate(dataloader))
            for i, (X, y) in t:
                
                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                outputs = net(X)
                loss = self.criterion(outputs, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if (i+1) % 2 == 0:
                    message = ('Epoch [{}/{}], Step [{}/{}], AvgLoss: {:.4f}'
                           .format(epoch+1, epochs, i+1, total_step, epoch_loss/N))
                    if self.config['general']['verbose']:
                        logger(message)
                    else:
                        t.set_description(message)
                    if self.writer: self.writer.add_scalar('data/loss', loss.item(), n_iter)
                    n_iter += 1
            if epoch % self.config['train']['save_every_epoch'] == 0:
                model.save()
        model.save()
        if self.writer: self.writer.close()

    def __str__(self):
        return 'segmentation trainer'

