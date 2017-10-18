from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from naive_bayes import nb

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
        default=1/nb.vocab_size,
        help='Beta for MAP')

    return vars(parser.parse_args())
