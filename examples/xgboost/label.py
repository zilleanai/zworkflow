import os
import numpy as np
import pandas as pd
from zworkflow.label import LabelBase


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
            df['action'] = df['Close'].shift(self.config['label']['forecast'])
            df['action'] = df['action'].pct_change()
            df.to_csv(os.path.join(datapath, f), index=False)

    def __str__(self):
        return 'honchar labeler'
