# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
import io
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from .predictbase import PredictBase


class ProbabilisticProgrammingPredict(PredictBase):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config['train']['device'])

    def predict(self, dataset, model):

        csv = None
        m = model.model()
        m.to(self.device)
        model.load()

        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'],
                shuffle=False, num_workers=4)

        predicted = []
        with torch.no_grad():
            for _, (X, y) in enumerate(dataloader):
                y = m(X).cpu().numpy()
                predicted.extend(y)
        df = pd.DataFrame(
            np.array(predicted), columns=self.config['dataset']['labels'])
        csv = df.to_csv(index=False)
        return csv.encode()

    def __str__(self):
        return 'pp predict'
