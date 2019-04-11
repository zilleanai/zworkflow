import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modelbase import ModelBase


class NeuralNet(nn.Module):
    def __init__(self, input_size, window, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.window = window
        self.kernel = 5
        self.relu = nn.ReLU()

        w = input_size
        h = window
        self.conv1 = nn.Conv2d(1, 32, 1, padding=1)
        w = self.outputSize(w, 1, padding=1)
        h = self.outputSize(h, 1, padding=1)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        w *= 2
        h *= 2
        self.conv2 = nn.Conv2d(32, 64, self.kernel, padding=1)
        w = self.outputSize(w, self.kernel, padding=1)
        h = self.outputSize(h, self.kernel, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        w /= 2
        h /= 2
        self.conv3 = nn.Conv2d(64, 128, self.kernel, padding=1)
        w = self.outputSize(w, self.kernel, padding=1)
        h = self.outputSize(h, self.kernel, padding=1)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        w *= 2
        h *= 2
        self.conv4 = nn.Conv2d(128, 64, self.kernel, padding=1)
        w = self.outputSize(w, self.kernel, padding=1)
        h = self.outputSize(h, self.kernel, padding=1)
        self.output_size = w * h * 64

        self.fc1 = nn.Linear(self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, input_size)
        self.fc4 = nn.Linear(input_size, output_size)

    def outputSize(self, in_size, kernel_size, stride=1, padding=0):
        # source: http://cs231n.github.io/convolutional-networks/
        # W2 = (W1-F+2P)/S+1
        # H2 = (H1-F+2P)/S+1
        # source https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        return(output)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.upsample2(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.view(-1, self.output_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        # out = torch.squeeze(out)
        return x

    # source: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class CSV2DModel(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.window = config['dataset'].get('window') or 1
        self.labels = config['dataset']['labels']
        self.initNet()

    def initNet(self):
        self.__net = NeuralNet(
            len(self.features), self.window, len(self.labels))
        self.__net.apply(self.__net.init_weights)

    def net(self):
        return self.__net

    def __str__(self):
        return 'csv_model: ' + str(self.features) + ', ' + str(self.labels)

    def save(self):
        torch.save(self.__net.state_dict(), self.config['model']['savepath'])
        if self.config['general']['verbose']:
            print('saved model: ', self.config['model']['savepath'])

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            if not self.config['train']['device'] is 'cpu':
                self.__net.load_state_dict(torch.load(
                    self.config['model']['savepath'], map_location='cpu'))
            else:
                self.__net.load_state_dict(torch.load(
                    self.config['model']['savepath']))

            if self.config['general']['verbose']:
                print('loaded model: ', self.config['model']['savepath'])
