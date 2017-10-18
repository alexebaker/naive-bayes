from __future__ import print_function
from __future__ import unicode_literals

from naive_bayes import cli
from naive_bayes import nb


training_csv = './data/training.csv'
testing_csv = './data/testing.csv'
training_matrix_file = './data/training_matrix.npz'
testing_matrix_file = './data/testing_matrix.npz'
classification_file = './classification.csv'


def main():
    """Main entry point."""
    # Parse the command line arguments
    cli_args = cli.parse_args()

    # Parse the training and testing data file given from the cli arguments
    parsed_matrix = nb.get_parsed_matrix(training_csv, training_matrix_file)
    frequency_matrix = nb.get_frequency_matrix(parsed_matrix)
    likelihood_matrix = nb.get_likelihood_matrix(frequency_matrix, beta=cli_args.get('beta'))

    # the training data is not needed anymore after we have the likelihood matrix,
    # so we overwrite this matix with the testing matrix to save space in memory
    parsed_matrix = nb.get_parsed_matrix(testing_csv, testing_matrix_file)

    # The testing data isn't needed after it is classified, so we overwrite it again,
    # to help save memory.
    parsed_matrix = nb.get_classification(parsed_matrix, likelihood_matrix)

    # Write the classification to a file for submission
    nb.save_classification(parsed_matrix, classification_file)
    return


if __name__ == "__main__":
    main()
