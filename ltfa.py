#!/usr/bin/env python3

from pprint import pprint
from pprint import pformat

import sys
import logging

import ltfa.util
from ltfa.util import LtfaError
from ltfa.util import ConfigError


def main():
    args = ltfa.parse_args(sys.argv[1:])
    ltfa.run(args)


if __name__ == '__main__':
    try:
        # Use this for profiling:
        #  import cProfile
        #  cProfile.run('main()', filename='ltfa.profile', sort='cumtime')

        sys.exit(main())
    except LtfaError as e:
        logging.error(e)
        sys.exit(1)
