from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import numpy as np

from scipy.sparse import csr_matrix, save_npz, load_npz


#np.set_printoptions(threshold='nan')


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
            matrix = np.zeros((12000, 61189), dtype=np.int32)
        else:
            matrix = np.zeros((12000, 61190), dtype=np.int32)
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

    :rtype: scipy.sparse.csr_matrix
    :returns: The computed frequency matrix based on the parsed matrix.
    """
    frequency_matrix = np.zeros((20, 61189), dtype=np.int32)
    for row in range(frequency_matrix.shape[0]):
        group_rows = parsed_matrix[np.where(parsed_matrix[:, -1] == row+1)[0], :]
        frequency_matrix[row, :-1] = np.sum(group_rows[:, :-1], axis=0)
        frequency_matrix[row, -1] = group_rows.shape[0]
    return frequency_matrix


def get_likelihood_matrix(frequency_matrix, beta=1):
    """Computes the likelihood matrix based on the given frequency matrix.

    :type frequency_martix: scipy.parse.csr_matrix
    :param frequency_matrix: matrix made from totaling words given the class

    :rtype:
    :returns: The computed likelihood matrix based on the frequency matrix.
    """
    likelihood_matrix = np.zeros(frequency_matrix.shape, dtype=np.float64)

    total_words = np.sum(frequency_matrix[:, :-1])

    word_prob = np.zeros((1, frequency_matrix.shape[1]-1), dtype=np.int32)
    group_prob = np.zeros((1, frequency_matrix.shape[0]), dtype=np.int32)

    col_sums = np.sum(frequency_matrix, axis=0)
    row_sums = np.sum(frequency_matrix[:, :-1], axis=1)

    try:
        word_prob = col_sums[:-1] / total_words
        group_prob = row_sums / total_words
    except ZeroDivisionError:
        print ("No words in matrix")

    #sums[sums == 0] = 1  # don't divide by 0, divide by 1 instead
    likelihood_matrix = (frequency_matrix + beta) / (col_sums + frequency_matrix.shape[1])
    return (likelihood_matrix, word_prob, group_prob)


def save_classification(classification, classification_file):
    """Saves the classification from naive bayes to a file.

    :type classification: list
    :param classification: The classification output from the ID3 algorithm for the testing data.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    return
