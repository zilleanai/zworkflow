#!/usr/bin/env python3

import argparse
import os
import sys

from mlworkflow import Config
from mlworkflow.dataset import get_dataset
from mlworkflow.model import get_model
from mlworkflow.train import get_train


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', help="Path to config file.")

    args, _ = parser.parse_known_args()

    configfile = args.config or {}
    config = Config(configfile)

    dataset = get_dataset(config)
    print(dataset)
    model = get_model(config)
    print(model)
    train = get_train(config)
    print(train)
    train.train(dataset, model)

    sys.exit(0)


if __name__ == '__main__':
    main()
