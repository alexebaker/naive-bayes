from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


#np.set_printoptions(threshold='nan')


def parse_training_file():
    """Parses the data out of the data file and into a format used by naive bayes.

    :type training_data: File Object
    :param training_data: A file object from the cli to parse into a data structure.

    :type vocab: File Object
    :param vocab: A file object from the cli to parse into a data structure.

    :type newsgroups: File Object
    :param newsgroups: A file object from the cli to parse into a data structure.

    :rtype: dict
    :returns: A data structure with the parsed data from the data file.
    """
    training_matrix = np.zeros((12000, 61188), dtype=np.int32)
    row = 0
    for line in training_file.readlines():
        training_matrix[row, :] = map(int, line.split(','))
        row += 1
    print(training_matrix.shape)


#    train_data = {}
#    test_data = {}
#    vocab = {}
#    newsgroups = {}

#    for line in news_groups:
#        newsgroup_id, newsgroup = line.split(' ')
#        newsgroups[int(newsgroup_id)] = newsgroup

#    word_id = 1
#    for line in vocabulary.readlines():
#        vocab[word_id] = line
#        word_id += 1

#    for line in training_data:
#        ids = line.split(',')
#        document_id = int(ids[0])
#        newsgroup_id = int(ids[-1])

#        train_data[document_id] = {}
#        train_data[document_id]['class'] = newsgroup_id
#        train_data[document_id]['words'] = {}

#        for word_id, word_count in enumerate(ids[1:-1]):
#            train_data[document_id]['words'][word_id+1] = int(word_count)

#    for line in testing_data:
#        ids = line.split(',')
#        document_id = int(ids[0])
#
#        test_data[document_id] = {}
#        test_data[document_id]['words'] = {}

#        for word_id, word_count in enumerate(ids[1:]):
#            test_data[document_id]['words'][word_id+1] = int(word_count)

    return training_matrix


def save_classification(classification, classification_file):
    """Saves the classification from naive bayes to a file.

    :type classification: list
    :param classification: The classification output from the ID3 algorithm for the testing data.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    return
