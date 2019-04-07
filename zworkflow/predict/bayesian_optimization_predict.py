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


class BayesianOptimizationPredict(PredictBase):

    def __init__(self, config, preprocessing):
        super().__init__(config)
        self.preprocessing = preprocessing

    def predict(self, files, model):

        csv = None
        model.load()

        tables = []
        if type(files) is bytes:
            f = io.BytesIO(files)
            f.seek(0)
            df = pd.read_csv(f)
            float_cols = [c for c in df if df[c].dtype == np.float64]
            df[float_cols] = df[float_cols].astype(np.float32)
            df = df.dropna()
            tables.append(df)
        else:
            for f in files:
                df = pd.read_csv(os.path.join(f))

                float_cols = [c for c in df if df[c].dtype == np.float64]
                df[float_cols] = df[float_cols].astype(np.float32)
                if self.preprocessing:
                    df = self.preprocessing.process(df)
                df = df.dropna()
                tables.append(df)
        data = pd.concat(tables, axis=0, ignore_index=True)
        data = data.reindex()

        X = data[self.config['dataset']['features']].values

        y = model.f(X, **model.max['params'])
        df = pd.DataFrame(
            y, columns=self.config['dataset']['labels'])
        csv = df.to_csv(index=False)
        if type(files) is bytes:
            return csv.encode()
        else:
            return csv

    def __str__(self):
        return 'bayesian_optimization_predict'
