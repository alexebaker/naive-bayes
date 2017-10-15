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


def get_likelihood_matrix(frequency_matrix, beta=.000016342):
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
        group_prob = np.log(row_sums / total_words)
    except ZeroDivisionError:
        print ("No words in matrix")

    #sums[sums == 0] = 1  # don't divide by 0, divide by 1 instead
    likelihood_matrix = np.log((frequency_matrix + beta) / (row_sums + beta))
    return (likelihood_matrix, word_prob, group_prob)


def classify_naive_bayes_row(document_row, likelihood_matrix, group_prob):
    """Computes the newsgroup that the document.

    :type document_row: array
    :param document_row: unclassified document

    :type likelihood_matrix: matrix
    :param likelihood_matrix: a table of word probabilities given the newsgroup

    :type group_prob: array
    :param group_prob: the probability of a particular newsgroup

    :rtype:
    :returns: The classification of the document.
    """
    classify_matrix = np.multiply(likelihood_matrix, document_row)
    row_sums = np.sum(classify_matrix[:, :-1], axis=1)
    freq_prob= np.add(row_sums,group_prob)
    newsgroup = 0
    argmax = 0
    for i in range(len(freq_prob)):
        if freq_prob[i]>argmax:
            argmax=freq_prob[i]
            newsgroup=i+1

    return newsgroup


def get_likelihood_matrix2(frequency_matrix, beta=1):
    """Computes the likelihood matrix based on the given frequency matrix.

    :type frequency_martix: ndarray
    :param frequency_matrix: matrix made from totaling words given the class

    :rtype: ndarray
    :returns: The computed likelihood matrix based on the frequency matrix.
    """
    likelihood_matrix = np.zeros(frequency_matrix.shape, dtype=np.float64)

    total_words = np.sum(frequency_matrix[:, :-1])
    total_docs = np.sum(frequency_matrix[:, -1])

    likelihood_matrix[:, :-1] = (frequency_matrix[:, :-1] + beta) / (total_words + vocab_size)
    likelihood_matrix[:, -1] = frequency_matrix[:, -1] / total_docs
    return np.log(likelihood_matrix)
    
def get_likelihood_matrix3(frequency_matrix, beta=1):
    """Computes the likelihood matrix based on the given frequency matrix.

    :type frequency_martix: ndarray
    :param frequency_matrix: matrix made from totaling words given the class

    :rtype: ndarray
    :returns: The computed likelihood matrix based on the frequency matrix.
    """
    
    likelihood_matrix = np.zeros(frequency_matrix.shape, dtype=np.float64)
    
    for i in range(20):
        total_words = np.sum(frequency_matrix[i,:])
        likelihood_matrix[i, :-1] = (frequency_matrix[i, :-1]+beta) / (total_words+vocab_size)

    total_docs = np.sum(frequency_matrix[:, -1])

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
    test_matrix[test_matrix.nonzero()] = 1

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
