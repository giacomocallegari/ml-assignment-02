import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn import metrics


def load_data():
    """Loads the input data and the output targets from the specified path."""

    print("")
    print("*** DATA LOADING ***")

    # Path of the CSV files.
    path = "..\\data\\ocr\\"

    # Number of considered examples, for debug reasons. Use "None" to take the whole dataset.
    train_size = 2000
    test_size = 500

    # Load the input data.
    X_train = np.genfromtxt(path + "train-data.csv", delimiter=",")[:train_size]
    X_test = np.genfromtxt(path + "test-data.csv", delimiter=",")[:test_size]

    # Load the output targets.
    y_train = np.genfromtxt(path + "train-targets.csv", dtype="str")[:train_size]
    y_test = np.genfromtxt(path + "test-targets.csv", dtype="str")[:test_size]

    print("Number of training examples: ", len(X_train))
    print("Number of test examples: ", len(X_test))

    return X_train, X_test, y_train, y_test


def cross_validation(X_train, y_train):
    """Performs k-fold cross validation to tune the model's parameters."""

    print("")
    print("*** CROSS VALIDATION ***")

    # Define the cross validation process.
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Specify the candidate values.
    gamma_values = [0.1, 0.05, 0.02, 0.01]
    accuracy_scores = []

    # Perform cross validation for each candidate value.
    for gamma in gamma_values:

        # Train the classifier.
        clf = SVC(C=10, kernel='rbf', gamma=gamma)

        # Compute the cross validation scores.
        scores = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='accuracy')

        # Compute the mean accuracy and save it.
        accuracy_score = scores.mean()
        accuracy_scores.append(accuracy_score)

        print("gamma = ", gamma, " \tscore = ", accuracy_score)

    # Get the gamma with highest mean accuracy.
    best_index = np.array(accuracy_scores).argmax()
    best_gamma = gamma_values[best_index]
    print("The best gamma is ", best_gamma)

    return best_gamma


def train(X_train, y_train, gamma):
    """Fits the classifier to the training set."""

    print("")
    print("*** TRAINING ***")

    clf = SVC(C=10, kernel='rbf', gamma=gamma)
    clf.fit(X_train, y_train)

    return clf


def test(clf, X_test, y_test):
    """Evaluates the classifier on the test set."""

    print("")
    print("*** TESTING ***")

    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    print("The testing accuracy is ", accuracy)


def main():
    """Main function."""

    # Load the data.
    X_train, X_test, y_train, y_test = load_data()

    # Perform cross validation to obtain the optimal parameters.
    gamma = cross_validation(X_train, y_train)

    # Train the optimal classifier.
    clf = train(X_train, y_train, gamma)

    # Test the classifier.
    test(clf, X_test, y_test)

    # Draw the learning curve.


# Start the program.
main()