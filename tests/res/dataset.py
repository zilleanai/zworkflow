import os
import numpy as np
import pandas as pd
from mlworkflow.dataset import DataSetBase


class dataset(DataSetBase):

    def __init__(self, config, preprocessing=None):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.preprocessing = preprocessing
        self.load(config['dataset']['datapath'])

    def load(self, datapath='.'):
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
        self.data = pd.concat(tables, axis=0, ignore_index=True)
        self.data = self.data.reindex()

    def __getitem__(self, idx):
        return self.data[self.features].loc[idx].values, self.data[self.labels].loc[idx].astype(np.float32).values

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "features: " + str(self.features) + " labels: " + str(self.labels) + " rows: " + str(len(self.data)) + str(self.files)
