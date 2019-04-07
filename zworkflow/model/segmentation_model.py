import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modelbase import ModelBase

# source: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py


def double_conv(in_channels, out_channels, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel, padding=1),
        nn.ReLU(inplace=True)
    )


class CNet(nn.Module):
    # source: https://github.com/chriamue/Cnet/blob/master/Cnet/Cnet.py
    # https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
    def __init__(self, classes):
        super(CNet, self).__init__()
        self.kernel = 3
        self.activation = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(64)

        self.down1 = double_conv(3, 64, self.kernel)
        self.down2 = double_conv(64, 128, self.kernel)
        self.down3 = double_conv(128, 256, self.kernel)
        self.down4 = double_conv(256, 512, self.kernel)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.up4 = double_conv(256+512, 256, self.kernel)
        self.up3 = double_conv(128+256, 128, self.kernel)
        self.up2 = double_conv(128+64, 64, self.kernel)
        self.up1 = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        x = self.down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.up2(x)
        out = self.up1(x)
        return out

    # source: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class SegmentationModel(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.width = config['dataset']['width']
        self.height = config['dataset']['height']
        self.outwidth = config['dataset']['outwidth']
        self.outheight = config['dataset']['outheight']
        self.classes = config['dataset']['classes']
        self.initNet()

    def initNet(self):
        self.__net = CNet(self.classes)
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
