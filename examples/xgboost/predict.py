# source: https://xgboost.readthedocs.io/en/latest/python/python_intro.html#
import os
import io
import numpy as np
import pandas as pd
import xgboost as xgb
from zworkflow.predict import PredictBase


class predict(PredictBase):

    def __init__(self, config):
        super().__init__(config)

    def predict(self, dataset, model):
        model.load()
        dtest = xgb.DMatrix(dataset.data(), label=dataset.label())
        predicted = model.model().predict(dtest)
        df = pd.DataFrame(
            np.array(predicted), columns=self.config['dataset']['labels'])
        csv = df.to_csv(index=False)
        return csv.encode()


    def __str__(self):
        return 'xgboost predict'
