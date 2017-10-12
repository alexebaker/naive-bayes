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
        type=int,
        default=1,
        help='Beta for MLE')

    return vars(parser.parse_args())
