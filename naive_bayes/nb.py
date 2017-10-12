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
        matrix = load_npz(matrix_file)
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
        matrix = csr_matrix(matrix)
        save_npz(matrix_file, matrix)
    return matrix


def get_frequency_matrix(parsed_matrix):
    """Computes the frequency matrix based on the given parsed matrix.

    :type parsed_martix: scipy.parse.csr_matrix
    :param parsed_matrix: matrix parsed from csv file

    :rtype: scipy.sparse.csr_matrix
    :returns: The computed frequency matrix based on the parsed matrix.
    """
    frequency_matrix=np.zeros((21, 61190), dtype=np.int32)
    return frequency_matrix


def get_likelihood_matrix(frequency_matrix):
    """Computes the likelihood matrix based on the given frequency matrix.

    :type frequency_martix: scipy.parse.csr_matrix
    :param frequency_matrix: matrix made from totaling words given the class

    :rtype:
    :returns: The computed likelihood matrix based on the frequency matrix.
    """
    """ Will add a row to the bottom for the count of each word divided by
    the count of all words and a will add a column to the end for the word
    count of that class divided by the total number of words."""
    likelihood_matrix = frequency_matrix[:, 1:-1]
    total_words=likelihood_matrix.sum()
    word_prob = []
    group_prob =[]
    sums = np.sum(likelihood_matrix, axis=0)
    row_sums=np.sum(likelihood_matrix, axis=1)
    try:
      for i in range(61190):
         word_prob.append(sums[i]/total_words)
      for j in range(21):
         group_prob.append(row_sums[j]/total_words)
    except ZeroDivisionError:
        print ("No words in matrix")
    sums[sums == 0] = 1  # don't divide by 0, divide by 1 instead
    likelihood_matrix = likelihood_matrix / sums
    return likelihood_matrix


def save_classification(classification, classification_file):
    """Saves the classification from naive bayes to a file.

    :type classification: list
    :param classification: The classification output from the ID3 algorithm for the testing data.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    return
