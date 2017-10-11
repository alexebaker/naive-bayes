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
        '--vocabulary',
        type=argparse.FileType(mode='r'),
        default='./data/vocabulary.txt',
        help='Path to the vocabulary file.')

    parser.add_argument(
        '--news-groups',
        type=argparse.FileType(mode='r'),
        default='./data/newsgrouplabels.txt',
        help='Path to the news group label files.')

    parser.add_argument(
        '--classification-file',
        type=argparse.FileType(mode='w'),
        default='./classification.csv',
        help='Path to the classification file to write the results of the testing data.')

    return vars(parser.parse_args())
