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

from .predictbase import PredictBase


class SegmentationPredict(PredictBase):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config['train']['device'])

    def predict(self, dataset, model):
        net = model.net()
        net.to(self.device)
        model.load()
        net.eval()
        dataloader = DataLoader(dataset, batch_size=1,
                shuffle=False, num_workers=4)

        predicted = []
        with torch.no_grad():
            for _, X in enumerate(dataloader):
                X = X.to(self.device)
                y = net(X).cpu().numpy()
                im = y
                im = im * 255
                im = im.astype(np.uint8)[0]
                im = np.squeeze(im)
                im = Image.fromarray(im)
                img_io = io.BytesIO()
                im.save(img_io, format='PNG')
                predicted.append(img_io)
        return predicted[0].getvalue()

    def __str__(self):
        return 'segmentation predict'

