# source: https://www.kaggle.com/youhanlee/simple-quant-features-using-python
import numpy as np
import pandas as pd
from zworkflow.preprocessing import PreprocessingBase


class preprocessing(PreprocessingBase):
    def __init__(self, config):
        super().__init__(config)
        self.functions = {
            'pct_change': self.pct_change
        }

    def pct_change(self, data):
        data[self.config['dataset']['features']] = data[self.config['dataset']['features']].pct_change()
        return data

    def process(self, data, verbose=False):
        data[self.config['dataset']['features']] = data[self.config['dataset']['features']].apply(pd.to_numeric, errors='coerce')
        for f in self.config['preprocessing']['functions']:
            fun = self.functions[f]
            data = fun(data)
        float_cols = [c for c in data if data[c].dtype == np.float64]
        data[float_cols] = data[float_cols].astype(np.float32)

        return data

    def __str__(self):
        return "preprocessing: " + str(list(self.functions.keys()))
