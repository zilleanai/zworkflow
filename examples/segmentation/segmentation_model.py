import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from zworkflow.model import ModelBase


class CNet(nn.Module):
    # source: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
    def __init__(self, outwidth, outheight, classes):
        super(CNet, self).__init__()
        self.classes = classes
        self.encoder = models.vgg16(pretrained=True)
        # freeze encoder
        for param in self.encoder.features.parameters():
            param.requires_grad = False
        self.kernel = 3
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.c1 = nn.Conv2d(512, 64, self.kernel)
        self.c2 = nn.Conv2d(64, 32, 1)
        self.out_size = 32*5*5
        self.outwidth = outwidth
        self.outheight = outheight
        self.fc1 = nn.Linear(self.out_size, outwidth*outheight*classes)

    def decoder(self, x):
        out = self.c1(x)
        out = self.activation(out)
        out = self.c2(out)
        out = out.view(-1, self.out_size)
        out = self.fc1(out)
        out = self.activation(out)
        out = out.view(-1, self.outwidth, self.outheight, self.classes)
        return out

    def forward(self, x):
        out = self.encoder.features(x)
        out = self.decoder(out)
        return out

    # source: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class model(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.width = config['dataset']['width']
        self.height = config['dataset']['height']
        self.outwidth = config['dataset']['outwidth']
        self.outheight = config['dataset']['outheight']
        self.classes = config['dataset']['classes']
        self.initNet()

    def initNet(self):
        self.__net = CNet(self.outwidth, self.outheight, self.classes)
        self.__net.apply(self.__net.init_weights)

    def net(self):
        return self.__net

    def __str__(self):
        return str(self.classes)

    def save(self):
        torch.save(self.__net.state_dict(), self.config['model']['savepath'])
        if self.config['general']['verbose']:
            print('saved model: ', self.config['model']['savepath'])

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            self.__net.load_state_dict(torch.load(
                self.config['model']['savepath']))
            if self.config['general']['verbose']:
                print('loaded model: ', self.config['model']['savepath'])
