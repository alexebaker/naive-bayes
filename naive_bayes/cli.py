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
        '--training-data',
        type=argparse.FileType(mode='r'),
        default='./data/training.csv',
        help='Path to the training data file.')

    parser.add_argument(
        '--testing-data',
        type=argparse.FileType(mode='r'),
        default='./data/testing.csv',
        help='Path to the test data file.')

    parser.add_argument(
        '--classification-file',
        type=argparse.FileType(mode='w'),
        default='./classification.csv',
        help='Path to the classification file to write the results of the testing data.')

    parser.add_argument(
        '--beta',
        type=float,
        default=1/nb.vocab_size,
        help='Beta for MAP')

    parser.add_argument(
        '--confusion',
        action='store_true',
        help='Print out the confusion matrix')

    return vars(parser.parse_args())
