import os
import io
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from zworkflow.evaluate import EvaluateBase

# source: https://github.com/usuyama/pytorch-unet/blob/master/loss.py
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 2 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class evaluate(EvaluateBase):

    def __init__(self, config):
        super().__init__(config)
        self.criterion = dice_loss or nn.CrossEntropyLoss().to(self.device)
        self.device = torch.device(self.config['train']['device'])

    def evaluate(self, dataset, model):
        model.load()
        net = model.net()
        net.to(self.device)
        net.eval()

        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'],
                                shuffle=False, num_workers=4)
        
        epoch_loss = 0.0
        losses = []
        t = tqdm(enumerate(dataloader))
        for i, (X, y) in t:
                
            X = X.to(self.device)
            y = y.to(self.device)

            outputs = net(X)
            loss = self.criterion(outputs, y)
            losses.append(loss.item())
            epoch_loss += loss.item()
        plt.figure()
        x = np.arange(len(losses))
        plt.plot(x, losses)
        plt.savefig('dice.png')
        plt.show()
        return { 'loss': epoch_loss, 'images': ['dice.png'] }

    def __str__(self):
        return 'segmentation evaluate'
