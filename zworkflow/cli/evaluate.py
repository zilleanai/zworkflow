#!/usr/bin/env python3

import argparse
import os
import select
import sys

from zworkflow import Config
from zworkflow.dataset import get_dataset
from zworkflow.preprocessing import get_preprocessing
from zworkflow.model import get_model
from zworkflow.evaluate import get_evaluate


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', help="Path to config file.")
    parser.add_argument("files",nargs="*")
    parser.add_argument('-v', '--verbose', action='store_true')

    args, _ = parser.parse_known_args()
    data = None
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        data = sys.stdin.buffer.read()

    configfile = args.config or {}
    config = Config(configfile)
    config['general']['verbose'] = args.verbose

    preprocessing = get_preprocessing(config)
    dataset = get_dataset(config, preprocessing, data=data or args.files)
    if config['general']['verbose']:
        print(dataset)
    model = get_model(config)
    if config['general']['verbose']:
        print(model)
    evaluate = get_evaluate(config)
    if config['general']['verbose']:
        print(evaluate)
    evaluate.evaluate(dataset, model)

    sys.exit(0)


if __name__ == '__main__':
    main()
