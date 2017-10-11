from __future__ import print_function
from __future__ import unicode_literals

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

    :rtype:
    :returns: The computed frequency matrix based on the parsed matrix.
    """
    # Ignore the first and last part of the matrix
    counts = parsed_matrix[:, 1:-1]

    #### Implemenataion needed
    frequency_matrix = np.zeros((20, 61188), dtype=np.int32)
    frequency_matrix = csr_matrix(frequency_matrix)
    return frequency_matrix


def save_classification(classification, classification_file):
    """Saves the classification from naive bayes to a file.

    :type classification: list
    :param classification: The classification output from the ID3 algorithm for the testing data.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    return
