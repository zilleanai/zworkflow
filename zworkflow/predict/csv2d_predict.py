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


class CSV2DPredict(PredictBase):

    def __init__(self, config, preprocessing):
        super().__init__(config)
        self.preprocessing = preprocessing
        self.device = torch.device(self.config['train']['device'])

    def pd_to_np_list(self, df, window=2):
        range_ = range
        if self.config['general'].get('verbose'):
            range_ = lambda a: tqdm(range(a))
            print(self.__str__(), 'data len: ', len(df), ', window: ', window)
        data_features = df[self.config['dataset']['features']]
        data = []
        for i in range_(len(df)-window):
            part = data_features[i:i+window]
            part = part.fillna(part.mean())
            part = part.astype(np.float32).values
            data.append(part)
        return data

    def predict(self, files, model):

        csv = None
        net = model.net()
        net.to(self.device)
        model.load()

        tables = []
        if type(files) is bytes:
            f = io.BytesIO(files)
            f.seek(0)
            df = pd.read_csv(f)
            float_cols = [c for c in df if df[c].dtype == np.float64]
            df[float_cols] = df[float_cols].astype(np.float32)
            if self.preprocessing:
                df = self.preprocessing.process(df)
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
        data_windows = self.pd_to_np_list(data[self.config['dataset']['features']], self.config['dataset'].get('window') or 1)

        X = np.atleast_3d(np.stack(data_windows))
        X = np.expand_dims(X, axis=3)
        X = np.rollaxis(X, 3, 1)
        X = torch.from_numpy(X)
        X = X.to(self.device)

        with torch.no_grad():
            y = net(X)
            y = y.cpu()
            df = pd.DataFrame(
                y.numpy(), columns=self.config['dataset']['labels'])
            csv = df.to_csv(index=False)
        if type(files) is bytes:
            return csv.encode()
        else:
            return csv

    def __str__(self):
        return 'csv2d_predict'
