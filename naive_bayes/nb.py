from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import numpy as np
import operator
import math

from scipy.sparse import csr_matrix, save_npz, load_npz


num_newsgroups = 20
num_docs = 12000
num_tests = 6774
vocab_size = 61188


def get_parsed_matrix(csv_file, matrix_file):
    """Parses the data out of the data file and into a format used by naive bayes.

    :rtype: ndarray
    :returns: A matrix read from the csv file
    """
    matrix = None

    # Check to see if we have a saved npz file first
    if os.path.isfile(matrix_file):
        sparse_matrix = load_npz(matrix_file)
        matrix = sparse_matrix.todense()
    else:
        # initialize the matrix with 0's. This will help speed up the time to parse the data file
        if 'testing' in csv_file.name:
            matrix = np.zeros((num_tests, vocab_size+1), dtype=np.int32)
        else:
            matrix = np.zeros((num_docs, vocab_size+2), dtype=np.int32)
        row = 0
        for line in csv_file.readlines():
            matrix[row, :] = map(int, line.split(','))
            row += 1

        # Gets rid of the first column of ids. We don't need this
        # since that information is based on the row. i.e. row 0 is ID 1.
        matrix = matrix[:, 1:]

        # save a sparse version of the matrix to reduce size and speed up reading time
        sparse_matrix = csr_matrix(matrix)
        save_npz(matrix_file, sparse_matrix)

    # returns a normal matrix. Sparse matrices don't have the same indexing power
    # as normal matrices, so we will be using normal matrices in the other functions.
    return matrix


def get_frequency_matrix(parsed_matrix):
    """Computes the frequency matrix based on the given parsed matrix.

    The frequency_matrix is a matrix counts how many of each word occured for each class.
    In this matrix, the row is the newsgroup and the col is the word. The last column is the
    count of how many documents were classified as that newsgroup.

    :type parsed_martix: numpy.ndarray
    :param parsed_matrix: matrix parsed from csv file

    :rtype: numpy.ndarray
    :returns: The computed frequency matrix based on the parsed matrix.
    """
    # Initialize the matrix with 0's. We add is an extra column to account for the counts
    # of the classes as well as the words.
    frequency_matrix = np.zeros((num_newsgroups, vocab_size+1), dtype=np.int32)
    for row in range(frequency_matrix.shape[0]):
        # this next line selects all rows, where the last column in that row is equal to the row_index + 1.
        # We add 1 since the matrix is 0 index but the document ids start at 1. When this operation completes,
        # group_rows is a matrix where the row is all the documents of the same news groups, and the cols are
        # the word counts for that document
        group_rows = parsed_matrix[np.where(parsed_matrix[:, -1] == row+1)[0], :]

        # This computes a column sum of the selected rows from the previous line. This will return a vector
        # which represents the total counts of a word for a given newsgroup. We don't sum the last column
        # since that is the newsgroup id
        frequency_matrix[row, :-1] = np.sum(group_rows[:, :-1], axis=0)

        # The last col in the frequency matrix is how many of each newsgroup there are.
        # Since group_rows are all rows that are equivalent to that row, we just need to select
        # the number of rows in this matrix for the count
        frequency_matrix[row, -1] = group_rows.shape[0]
    return frequency_matrix


def get_likelihood_matrix(frequency_matrix, beta=1/vocab_size, ranked=True, num_ranked_words=100):
    """Computes the likelihood matrix based on the given frequency matrix.

    The likelihood_matrix is all of the conditional probabilities needed for naive bayes.
    each entry is the probability of the word (col) given the class (row). The last col
    is the MLE probability for each newsgroup.

    :type frequency_martix: numpy.ndarray
    :param frequency_matrix: matrix made from totaling words given the class

    :type beta: float
    :param beta: beta used in MAP calculations

    :rtype: numpy.ndarray
    :returns: The computed likelihood matrix based on the frequency matrix.
    """
    # Initialize this matrix with 0's. It will be the same size as the frequency matrix
    likelihood_matrix = np.zeros(frequency_matrix.shape, dtype=np.float64)

    # word counts is a row sum of the frequency matrix, i.e., the total number of words for a given class.
    # the last column is ignored since that is a count of the newsgroups and not the words.
    word_counts = np.sum(frequency_matrix[:, :-1], axis=1).reshape((frequency_matrix.shape[0], 1))

    # this is just a vector sum, since only the last column has counts for each newsgroup
    total_docs = np.sum(frequency_matrix[:, -1])

    if ranked:
        get_ranked_list(word_counts, frequency_matrix, num_ranked_words)

    # This line computes the MAP probability based on the frequency matrix and beta
    # This ignores the last col since that is the counts of the newsgroups, and not the words.
    likelihood_matrix[:, :-1] = (frequency_matrix[:, :-1] + beta) / (word_counts + (beta * vocab_size))

    # This line computes the MLE probability of each newsgroups for the last column.
    likelihood_matrix[:, -1] = frequency_matrix[:, -1] / total_docs

    # We return the log of the likihood matrix for future calculations
    return np.log(likelihood_matrix)

def get_ranked_list(word_counts, frequency_matrix, num_ranked_words=100):
    """Computes highest ranked words after filtering rare words and very common words like 'the'.

    :type word_counts: numpy.ndarray that is in fact a 1D array
    :param word_counts: the total number of words for a given class.

    :type frequency_martix: numpy.ndarray
    :param frequency_matrix: matrix made from totaling words given the class

    :type num_ranked_words: int32
    :param num_ranked_words: the number of words to be printed that the classifier relies on.
    """

    #Calculate the mean
    mean = np.sum(word_counts[:,-1])/vocab_size
    col_sums = np.sum(frequency_matrix[:,:], axis=0)

    #Subtract the mean from each data point and sqaure each difference
    sq_diff= (col_sums[:]-mean)**2

    #Calculate the mean of the square differences, then take the square root.
    std_dev = math.sqrt(np.sum(sq_diff[:])/vocab_size)

    #Produce a vector to zero out words that are 2 std deviations from the mean.
    for i in range(len(col_sums)-1):
        if col_sums[i]<(mean-(2*std_dev)) or col_sums[i]>(mean+(2*std_dev)):
            col_sums[i]=0
        else:
            col_sums[i]=1

    #new_freq_matrix only has values that are within 2 standard deviations of mean
    #This eliminates common words like "the" and rare words as well.
    new_freq_matrix = np.zeros(frequency_matrix.shape, dtype=np.float64)
    new_freq_matrix = np.multiply(frequency_matrix, col_sums)
    beta=1/vocab_size
    temp_matrix = np.zeros(new_freq_matrix.shape, dtype=np.float64)

    #This is the MAP, which contains a prior allowing us to take the log because there are no 0's
    temp_matrix[:, :-1] = (new_freq_matrix[:, :-1] + beta) / (word_counts + beta)
    temp_matrix[:, -1] = 1
    temp_matrix=np.log(temp_matrix)

    #each value in the matrix is multiplied by its probability, need for the information gain equation
    entropy_matrix = np.zeros(new_freq_matrix.shape, dtype=np.float64)
    entropy_matrix[:, :-1] = (new_freq_matrix[:, :-1]/word_counts)*temp_matrix[:, :-1]

    #values are summed but the sign is never flipped because the relationship of maximum remians.
    entropy_vector = np.sum(entropy_matrix[:-1, :], axis=0).reshape((1, entropy_matrix.shape[1]))

    #This list is sorted so the highest information gain is at the front, the most negative number.
    sorted_entropy = sorted(entropy_vector.transpose().tolist())

    #This grabs the element that is at the rank cutoff position
    min_entropy = sorted_entropy[num_ranked_words-1]

    #This opens vocabulary with read only permission.
    f = open('./data/vocabulary.txt', "r")

    #Each line is an element in a list and close f
    words = f.readlines()
    f.close()

    #Here the words that are going to be printed are paired with their information gain value.
    ranked_list=[]
    for i in range(entropy_vector.shape[1]-1):
        if entropy_vector[0][i]<=min_entropy:
            ranked_list.append([entropy_vector[0][i], words[i]])
    f1=open('./testfile.txt', 'w+')

    #This adds the highest ranked words to the document.  The default is 100
    sorted_ranked_list = sorted(ranked_list, key=operator.itemgetter(0))
    for i in range(len(sorted_ranked_list)):
        temp_string = "%d. %s" %(i+1, sorted_ranked_list[i][1])
        f1.write(temp_string)
    f1.close()


def get_confusion_matrix(parsed_matrix, beta=1/vocab_size):
    # break the training and testing matrix up by the first 11,000 docs
    # and the last 1,000 docs
    training_matrix = parsed_matrix[0:11000, :]
    testing_matrix = parsed_matrix[11000:, :]

    frequency_matrix = get_frequency_matrix(training_matrix)
    likelihood_matrix = get_likelihood_matrix(frequency_matrix, beta=beta)

    classification = get_classification(testing_matrix[:, :-1], likelihood_matrix)

    real_class = testing_matrix[:, -1]
    calc_class = classification[:, 1].reshape((real_class.size, 1))

    # this tests which columns are equal (i.e. the calculated class = the acutal class)
    # and divides that number by the total number of test documents
    accuracy = np.sum(real_class == calc_class) / real_class.size

    confusion_matrix = np.zeros((num_newsgroups, num_newsgroups), dtype=np.int32)
    for row in range(num_newsgroups):
        for col in range(num_newsgroups):
            # this computes the confusion matrix. First, is selects the documents
            # that are classified for the real group (column) and then compares it
            # to the calculated groups for that class (row)
            confusion_matrix[row][col] = np.sum(calc_class[real_class == col+1] == row+1)

    # this is required to print out an entire matrix, and not just the first few columns and rows.
    np.set_printoptions(threshold='nan')

    # print out accuracy and confusion matrix
    print('accuracy = %f' % accuracy)
    print(confusion_matrix)
    return


def get_classification(test_matrix, likelihood_matrix, starting_id=12001):
    """Computes the classification of the test data given the frequency_matrix.

    :type test_martix: numpy.ndarray
    :param test_matrix: matrix read from a csv file to classify

    :type frequency_martix: numpy.ndarray
    :param frequency_matrix: matrix made from totaling words given the class

    :rtype: numpy.ndarray
    :returns: A mapping of document id in the testing data to newsgroup id.
    """
    # the classification matrix has two columns, the first is the document id,
    # and the second column is the class.
    classification = np.zeros((test_matrix.shape[0], 2), dtype=np.int32)

    # the test data starts at document id 12001, so we add that to
    # the first column.
    classification[:, 0] = np.arange(test_matrix.shape[0]) + starting_id

    # add a col of 1's as the last column of the test matrix. this is to
    # always count the MLE probability in the matrix multiplication.
    tmp = np.ones((test_matrix.shape[0], test_matrix.shape[1]+1))
    tmp[:, :-1] = test_matrix
    test_matrix = tmp

    # to compute the liklihood of each document for a class, we do the matrix
    # multiplication of the liklihood matrix with the transpose of the test_matrix.
    # this ensures that the matrix dimension line up. When the matrix multiplication is done,
    # the product matrix is a matrix where the row is the newsgroup, and the column is the document to,
    # classify.
    product = likelihood_matrix.dot(test_matrix.T)

    # To get the final classification, we do argmax on the column, which will return the
    # index of the row (newsgroup) with the largest probability. We add 1 to this value,
    # since the matrix is 0 indexed and the group ids start at 1.
    classification[:, 1] = np.argmax(product, axis=0) + 1
    return classification


def save_classification(classification, classification_file):
    """Saves the classification from naive bayes to a file.

    :type classification: list
    :param classification: The classification output from the ID3 algorithm for the testing data.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    print("id,class", file=classification_file)
    for row in classification:
        print("%d,%d" % (row[0], row[1]), file=classification_file)
    return
