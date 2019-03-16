# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
import io
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from mlworkflow.predict import PredictBase


class predict(PredictBase):

    def __init__(self, config, preprocessing):
        super().__init__(config)
        self.preprocessing = preprocessing
        self.device = torch.device(self.config['train']['device'])

    def predict(self, files, model):

        csv = None
        net = model.net()
        net.to(self.device)
        model.load()
        files = 'train/9_4.tif'
        data = []
        

        image = Image.open(files)
        image = image.convert('RGB')
        image = np.array(image)
        image = cv2.resize(
            image, (self.config['dataset']['height'],self.config['dataset']['width']))
        image = np.rollaxis(image, 2, 0)
        image = image.astype(np.float32)
        X = np.expand_dims(image, axis=0)
        X = torch.from_numpy(X)
        X = X.to(self.device)
        with torch.no_grad():
            y = net(X)
            y = y.cpu()
        im = y.numpy()
        im = im * 255
        im = im.astype(np.uint8)[0]
        im = np.squeeze(im)
        im = Image.fromarray(im)
        im.save("segmentation_out.jpeg")

    def __str__(self):
        return 'my predict'
