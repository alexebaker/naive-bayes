from __future__ import print_function
from __future__ import unicode_literals

import argparse


def parse_args():
    """Parse CLI arguments.

    :rtype: dict
    :returns: Dictonairy of parsed cli arguments.
    """

    # argument parser object
    parser = argparse.ArgumentParser(
        description='Classifies the testing data using naive bayes and the training data.')

    # Add arguments to the parser
    parser.add_argument(
        '--beta',
        type=float,
        default=1/61188,
        help='Beta for MAP')

    return vars(parser.parse_args())
