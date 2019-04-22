# source: https://xgboost.readthedocs.io/en/latest/python/python_intro.html#
from zworkflow.model import ModelBase
import os
import xgboost as xgb

class model(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.__model = xgb.Booster({'nthread': config['model']['nthread']})

    def model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    def __str__(self):
        return 'xgboost model: ' + str(self.features) + ', ' + str(self.labels)

    def save(self):
        self.__model.save_model(
                   self.config['model']['savepath'])

        if self.config['general']['verbose']:
            print('saved model: ', self.config['model']['savepath'])

    def load(self):
        if os.path.exists(self.config['model']['savepath']):
            self.__model.load_model(self.config['model']['savepath'])
            if self.config['general']['verbose']:
                print('loaded model: ', self.config['model']['savepath'])
