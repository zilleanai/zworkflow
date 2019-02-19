import pytest
from mlworkflow import Config

config = {}

def test_defaults():
    config_ = Config(config=config)
    assert 'dataset' in config_.config
    assert 'model' in config_
    assert 'train' in config_
