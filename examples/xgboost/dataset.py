import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from zworkflow.dataset import DataSetBase


class dataset(DataSetBase):

    def __init__(self, config, preprocessing=None, data=None):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.preprocessing = preprocessing
        self.__load(config['dataset']['datapath'])

    def __load(self, datapath='.'):
        self.files = sorted([f for f in os.listdir(datapath)
                             if f.endswith('.csv') or f.endswith('.gz')])
        tables = []
        for f in self.files:
            df = pd.read_csv(os.path.join(datapath, f))
            float_cols = [c for c in df if df[c].dtype == np.float64]
            df[float_cols] = df[float_cols].astype(np.float32)
            if self.preprocessing:
                df = self.preprocessing.process(df)
            df = df.dropna()
            for label in self.labels:
                if not label in df:
                    df[label] = 0.0
            tables.append(df)
        self.__data = pd.concat(tables, axis=0, ignore_index=True)
        self.__data = self.__data.reindex()

    def data(self):
        return self.__data[self.features]

    def label(self):
        return self.__data[self.labels]

    def __getitem__(self, idx):
        X = self.__data[self.features][idx].astype(np.float32).values
        y = self.__data[self.labels][idx].astype(np.float32).values
        return X.flatten(), y.flatten()

    def __len__(self):
        return len(self.__data)

    def __str__(self):
        return "xgboost_dataset, features: " + str(self.features) + " labels: " + str(self.labels) + " rows: " + str(len(self.data)) + str(self.files)
