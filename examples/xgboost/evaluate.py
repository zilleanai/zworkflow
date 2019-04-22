# source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
import os
import io
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from zworkflow.evaluate import EvaluateBase


class evaluate(EvaluateBase):

    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, dataset, model):
        return { 'images': ['importance.png', 'tree.png'] }


    def __str__(self):
        return 'xgboost evaluate'
