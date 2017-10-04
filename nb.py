from __future__ import print_function
from __future__ import unicode_literals

from naive_bayes import cli
from naive_bayes import nb


def main():
    """Main entry point to the decision tree."""
    # Parse the command line arguments
    cli_args = cli.parse_args()

    # Parse the training and testing data file given from the cli arguments
    training_data = nb.parse_data(cli_args.get('training_data'))
    testing_data = nb.parse_data(cli_args.get('testing_data'))

    classification = None

    # Write the classification to a file for submission
    nb.save_classification(classification, cli_args.get('classification_file'))
    return


if __name__ == "__main__":
    main()
