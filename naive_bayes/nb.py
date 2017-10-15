from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import numpy as np

from scipy.sparse import csr_matrix, save_npz, load_npz


#np.set_printoptions(threshold='nan')


num_newsgroups = 20
num_docs = 12000
num_tests = 6774
vocab_size = 61188


def get_parsed_matrix(csv_file, matrix_file):
    """Parses the data out of the data file and into a format used by naive bayes.

    :rtype: scipy.sparse.csr_matrix
    :returns: A sparse matrix read from the csv file
    """
    matrix = None
    if os.path.isfile(matrix_file):
        sparse_matrix = load_npz(matrix_file)
        matrix = sparse_matrix.todense()
    else:
        if 'testing' in csv_file:
            matrix = np.zeros((num_tests, vocab_size+1), dtype=np.int32)
        else:
            matrix = np.zeros((num_docs, vocab_size+2), dtype=np.int32)
        row = 0
        with open(csv_file, 'r') as f:
            for line in f.readlines():
                matrix[row, :] = map(int, line.split(','))
                row += 1
        matrix = matrix[:, 1:]
        sparse_matrix = csr_matrix(matrix)
        save_npz(matrix_file, sparse_matrix)
    return matrix


def get_frequency_matrix(parsed_matrix):
    """Computes the frequency matrix based on the given parsed matrix.

    :type parsed_martix: scipy.parse.csr_matrix
    :param parsed_matrix: matrix parsed from csv file

    :rtype: ndarray
    :returns: The computed frequency matrix based on the parsed matrix.
    """
    frequency_matrix = np.zeros((num_newsgroups, vocab_size+1), dtype=np.int32)
    for row in range(frequency_matrix.shape[0]):
        group_rows = parsed_matrix[np.where(parsed_matrix[:, -1] == row+1)[0], :]
        frequency_matrix[row, :-1] = np.sum(group_rows[:, :-1], axis=0)
        frequency_matrix[row, -1] = group_rows.shape[0]
    return frequency_matrix


def get_likelihood_matrix(frequency_matrix, beta=1/vocab_size):
    """Computes the likelihood matrix based on the given frequency matrix.

    :type frequency_martix: ndarray
    :param frequency_matrix: matrix made from totaling words given the class

    :rtype: ndarray
    :returns: The computed likelihood matrix based on the frequency matrix.
    """

    likelihood_matrix = np.zeros(frequency_matrix.shape, dtype=np.float64)

    word_counts = np.sum(frequency_matrix[:, :-1], axis=1).reshape((frequency_matrix.shape[0], 1))
    total_docs = np.sum(frequency_matrix[:, -1])

    likelihood_matrix[:, :-1] = (frequency_matrix[:, :-1] + beta) / (word_counts + beta)
    likelihood_matrix[:, -1] = frequency_matrix[:, -1] / total_docs
    return np.log(likelihood_matrix)


def get_classification(test_matrix, likelihood_matrix):
    classification = np.zeros((test_matrix.shape[0], 2), dtype=np.int32)
    classification[:, 0] = np.arange(test_matrix.shape[0]) + 12001

    # add a col of 1's as the last col. This represents P(x)
    tmp = np.ones((test_matrix.shape[0], test_matrix.shape[1]+1))
    tmp[:, :-1] = test_matrix
    test_matrix = tmp

    # Set all of the counts to 1. idk how to deal with the counts with the logs
    # So just doing this for now
    #test_matrix[test_matrix.nonzero()] = 1

    product = likelihood_matrix.dot(test_matrix.T)
    classification[:, 1] = np.argmax(product, axis=0) + 1
    return classification


def save_classification(classification, classification_file):
    """Saves the classification from naive bayes to a file.

    :type classification: list
    :param classification: The classification output from the ID3 algorithm for the testing data.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    with open(classification_file, 'w') as f:
        print("id,class", file=f)
        for row in classification:
            print("%d,%d" % (row[0], row[1]), file=f)
    return
