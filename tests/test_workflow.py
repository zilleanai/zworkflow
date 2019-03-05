import os
import io
import pytest
from mlworkflow import Config
from mlworkflow.dataset import get_dataset
from mlworkflow.model import get_model
from mlworkflow.train import get_train
from mlworkflow.predict import get_predict
from mlworkflow.label import get_label

def train_logger(project, *args, **kwargs):
    mapped = "".join(map(str, args))
    print(mapped)


def test_label():
    os.chdir(os.path.join('tests','res'))
    configfile = os.path.join('workflow.yml') or {}
    config = Config(configfile)
    label = get_label(config)
    label.label()
    
    os.chdir(os.path.join('..','..'))

def test_train():
    os.chdir(os.path.join('tests','res'))
    configfile = os.path.join('workflow.yml') or {}
    config = Config(configfile)
    config['general']['verbose'] = True

    dataset = get_dataset(config)
    if config['general']['verbose']:
        print(dataset)
    model = get_model(config)
    if config['general']['verbose']:
        print(model)
    logger = lambda *args, **kwargs: train_logger('project', args, kwargs)
    train = get_train(config)
    if config['general']['verbose']:
        print(train)
    train.train(dataset, model, logger=logger)
    os.chdir(os.path.join('..','..'))

def test_predict():
    os.chdir(os.path.join('tests','res'))
    configfile = os.path.join('workflow.yml') or {}
    config = Config(configfile)
    config['general']['verbose'] = False

    model = get_model(config)
    if config['general']['verbose']:
        print(model)
    files = ['ETH-USDT_2019-02-28T18:00:00_2019-03-01T00:00:00.gz']
    predict = get_predict(config)
    print(predict.predict(files, model))
    os.chdir(os.path.join('..','..'))