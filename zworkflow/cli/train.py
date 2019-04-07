#!/usr/bin/env python3

import argparse
import os
import sys

from zworkflow import Config
from zworkflow.dataset import get_dataset
from zworkflow.preprocessing import get_preprocessing
from zworkflow.model import get_model
from zworkflow.train import get_train


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', help="Path to config file.")
    parser.add_argument('-v', '--verbose', action='store_true')

    args, _ = parser.parse_known_args()

    configfile = args.config or {}
    config = Config(configfile)
    config['general']['verbose'] = args.verbose

    preprocessing = get_preprocessing(config)
    dataset = get_dataset(config, preprocessing)
    if config['general']['verbose']:
        print(dataset)
    model = get_model(config)
    if config['general']['verbose']:
        print(model)
    train = get_train(config)
    if config['general']['verbose']:
        print(train)
    train.train(dataset, model)

    sys.exit(0)


if __name__ == '__main__':
    main()
