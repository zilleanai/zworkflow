#!/usr/bin/env python3

import argparse
import os
import sys

from zworkflow import Config
from zworkflow.label import get_label


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', help="Path to config file.")
    parser.add_argument('-v', '--verbose', action='store_true')

    args, _ = parser.parse_known_args()

    configfile = args.config or {}
    config = Config(configfile)
    config['general']['verbose'] = args.verbose

    label = get_label(config)
    label.label()

    sys.exit(0)


if __name__ == '__main__':
    main()
