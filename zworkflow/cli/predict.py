#!/usr/bin/env python3

import argparse
import os
import sys

from zworkflow import Config
from zworkflow.model import get_model
from zworkflow.preprocessing import get_preprocessing
from zworkflow.predict import get_predict


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', help="Path to config file.")
    parser.add_argument("files",nargs="*")
    parser.add_argument('-v', '--verbose', action='store_true')
    args, _ = parser.parse_known_args()

    configfile = args.config or {}
    config = Config(configfile)
    config['general']['verbose'] = args.verbose

    model = get_model(config)
    if config['general']['verbose']:
        print(model)
    preprocessing = get_preprocessing(config)
    predict = get_predict(config, preprocessing)
    print(predict.predict(args.files, model))

    sys.exit(0)


if __name__ == '__main__':
    main()
