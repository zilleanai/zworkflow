import io
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .datasetbase import DataSetBase


class CSV2DDataset(DataSetBase):

    def __init__(self, config, preprocessing=None, data=None):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.preprocessing = preprocessing
        self.load(config['dataset']['datapath'], data)

    def pd_to_np_list(self, df, window=2):
        range_ = range
        if self.config['general'].get('verbose'):
            def range_(a): return tqdm(range(a))
            print(self.__str__(), 'data len: ', len(df), ', window: ', window)
        data = []
        for i in range_(len(df)-window):
            part = df[i:i+window]
            data.append(part.fillna(part.mean()))
        return data

    def prepare(self, df):
        float_cols = [c for c in df if df[c].dtype == np.float64]
        df[float_cols] = df[float_cols].astype(np.float32)
        if self.preprocessing:
            df = self.preprocessing.process(df)
        df = df.dropna()
        for label in self.labels:
            if not label in df:
                df[label] = 0.0
        return df

    def load(self, datapath='.', data=None):
        tables = []
        if data:
            if type(data) is bytes:
                data = io.BytesIO(data)
                data.seek(0)
                df = pd.read_csv(data)
                df = self.prepare(df)
                tables.append(df)
            elif type(data) is list:
                self.files = data
                for f in self.files:
                    df = pd.read_csv(f)
                    df = self.prepare(df)
                    tables.append(df)
        else:
            self.files = sorted([f for f in os.listdir(datapath)
                                 if f.endswith('.csv') or f.endswith('.gz')])
            for f in self.files:
                df = pd.read_csv(os.path.join(datapath, f))
                df = self.prepare(df)
                tables.append(df)
        self.data = pd.concat(tables, axis=0, ignore_index=True)
        self.data = self.data.reindex()
        self.data_windows = self.pd_to_np_list(
            self.data[self.features], self.config['dataset'].get('window') or 1)

    def __getitem__(self, idx):
        image = np.atleast_3d(self.data_windows[idx].astype(np.float32).values)
        image = np.rollaxis(image, 2, 0)
        return image, self.data[self.labels].loc[idx].astype(np.float32).values

    def __len__(self):
        return len(self.data_windows)

    def __str__(self):
        return "csv2d_dataset, features: " + str(self.features) + " labels: " + str(self.labels) + " rows: " + str(len(self.data)) + str(self.files)
