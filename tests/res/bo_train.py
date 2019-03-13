# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from mlworkflow.train import TrainBase

from bayes_opt import BayesianOptimization

class train(TrainBase):

    def __init__(self, config):
        super().__init__(config)

    def train(self, dataset, model, logger=print):

        if self.config['train']['load_model']:
            model.load()

        epochs = self.config['train']['epochs']
        total_step = len(dataset) // self.config['train']['batch_size']
        dataloader = DataLoader(dataset, batch_size=self.config['train']['batch_size'],
                                shuffle=True, num_workers=4)

        for i, (X, y) in tqdm(enumerate(dataloader)):
            
            model.data = X.data.numpy()
            model.target = y.data.numpy()
            optimizer = BayesianOptimization(
                f=model.optimizable,
                pbounds=model.pbounds,
                verbose=2,
                random_state=20,
            )
            model.load_logs(optimizer)
            optimizer.maximize(
                init_points=10,
                n_iter=self.config['train']['epochs']
            )
            model.res = optimizer.res
            model.max = optimizer.max
        model.save()

    def __str__(self):
        return 'my trainer'
