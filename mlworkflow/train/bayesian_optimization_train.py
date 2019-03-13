import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization

from . import TrainBase


class BayesianOptimizationTrain(TrainBase):

    def __init__(self, config):
        super().__init__(config)

    def train(self, dataset, model, logger=print):

        if self.config['train']['load_model']:
            model.load()

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
        return 'bayesian_optimization_trainer'
