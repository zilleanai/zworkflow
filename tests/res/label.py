import os
import numpy as np
import pandas as pd
from mlworkflow.label import LabelBase


class label(LabelBase):

    def __init__(self, config):
        super().__init__(config)

    def label(self, verbose=False, logger=print):
        datapath = self.config['dataset']['datapath']
        files = sorted([f for f in os.listdir(datapath)
                        if f.endswith('.csv') or f.endswith('.gz')])

        for f in files:
            logger('label file: ', f)
            df = pd.read_csv(os.path.join(datapath, f))
            df['action'] = np.log(df['price'] / df['price'].shift(10))
            df.to_csv(os.path.join(datapath, f), compression='gzip')

    def __str__(self):
        return 'labeler'
