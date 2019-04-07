# source: https://www.kaggle.com/youhanlee/simple-quant-features-using-python
import numpy as np
import pandas as pd
from .preprocessingbase import PreprocessingBase


class CSVPreprocessing(PreprocessingBase):
    def __init__(self, config):
        super().__init__(config)
        self.functions = {
            'move': self.move,
            'moving_average10': self.moving_average10,
            'moving_average50': self.moving_average50,
            'exponential_moving_average10': self.exponential_moving_average10,
            'exponential_moving_average50': self.exponential_moving_average50,
            'macd': self.macd,
            'volume_moving_average10': self.volume_moving_average10
        }

    def move(self, data):
        data['move'] = data['price'] - data['price'].shift(10)
        return data

    def moving_average10(self, data):
        data['moving_average10'] = data['price'].rolling(window=10).mean()
        return data

    def moving_average50(self, data):
        data['moving_average50'] = data['price'].rolling(window=50).mean()
        return data

    def exponential_moving_average10(self, data):
        ewma = pd.Series.ewm
        data['exponential_moving_average10'] = ewma(
            data["price"], span=10).mean()
        return data

    def exponential_moving_average50(self, data):
        ewma = pd.Series.ewm
        data['exponential_moving_average50'] = ewma(
            data["price"], span=50).mean()
        return data

    def macd(self, data):
        data['macd'] = data['exponential_moving_average10'] - \
            data['exponential_moving_average50']
        return data

    def volume_moving_average10(self, data):
        data['volume_moving_average10'] = data['ask_quantity_0'].rolling(
            window=10).mean()
        return data

    def process(self, data, verbose=False):
        for f in self.config['preprocessing']['functions']:
            fun = self.functions[f]
            data = fun(data)
        float_cols = [c for c in data if data[c].dtype == np.float64]
        data[float_cols] = data[float_cols].astype(np.float32)

        return data

    def __str__(self):
        return "csv_preprocessing: " + str(list(self.functions.keys()))
