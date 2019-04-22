import os
import matplotlib.pyplot as plt
import xgboost as xgb
from zworkflow.train import TrainBase


class train(TrainBase):

    def __init__(self, config):
        super().__init__(config)

    def train(self, dataset, model, logger=print):
        dtrain = xgb.DMatrix(dataset.data(), label=dataset.label())
        param = self.config['train'].get('param') or {}
        num_round = self.config['train']['epochs']
        evallist = [(dtrain, 'train')]
        model.set_model(xgb.train(param, dtrain, num_round, evallist))
        model.save()
        xgb.plot_importance(model.model())
        plt.savefig('importance.png')
        xgb.plot_tree(model.model(), num_trees=2)
        plt.savefig('tree.png')

    def __str__(self):
        return 'xgboost trainer'
