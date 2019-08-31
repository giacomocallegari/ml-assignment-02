import numpy as np


def load_data():
    """Loads the input data and the output targets from the specified path."""

    # Path of the data.
    path = "..\\data\\ocr\\"

    # Load the input data.
    print("Loading the input data...")
    X_train = np.genfromtxt(path + "train-data.csv", delimiter=",")
    X_test = np.genfromtxt(path + "test-data.csv", delimiter=",")

    # Load the output targets.
    print("Loading the output targets...")
    y_train = np.genfromtxt(path + "train-targets.csv", dtype="str")
    y_test = np.genfromtxt(path + "test-targets.csv", dtype="str")

    print("Number of training examples: ", len(X_train))
    print("Number of test examples: ", len(X_test))

    return X_train, X_test, y_train, y_test


def main():
    """Main function."""

    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Perform cross validation

    # Train the optimal classifier

    # Test the classifier

    # Draw the learning curve


# Start the program
main()