import os
import pandas as pd
from mlworkflow.dataset import DataSetBase


class dataset(DataSetBase):

    def __init__(self, config):
        super().__init__(config)
        self.load(config['dataset']['datapath'])

    def load(self, datapath='.'):
        self.files = sorted([f for f in os.listdir(datapath)
                             if f.endswith('.csv') or f.endswith('.gz')])
        tables = []
        for f in self.files:
            df = pd.read_csv(os.path.join(datapath, f))
            tables.append(df)
        self.data = pd.concat(tables, axis=0, ignore_index=True)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def __str__(self):
        return str(self.files) + " rows: " + str(len(self.data))
