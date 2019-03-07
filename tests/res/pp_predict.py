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

from mlworkflow.predict import PredictBase


class predict(PredictBase):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config['train']['device'])

    def predict(self, files, model):

        csv = None
        m = model.model()
        m.to(self.device)
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
                df = df.dropna()
                tables.append(df)
        data = pd.concat(tables, axis=0, ignore_index=True)
        data = data.reindex()

        X = data[self.config['dataset']['features']].values
        X = torch.from_numpy(X)
        X = X.to(self.device)

        with torch.no_grad():
            y = net(X)
            y = y.cpu()
            #y = np.squeeze(y)
            print(y)
            df = pd.DataFrame(
                y.numpy(), columns=self.config['dataset']['labels'])
            csv = df.to_csv(index=False)
        if type(files) is bytes:
            return csv.encode()
        else:
            return csv

    def __str__(self):
        return 'pp predict'
