import numpy as np
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt


def load_data():
    """Loads the input data and the output targets from the specified path."""

    print("")
    print("*** DATA LOADING ***")

    # Path of the CSV files.
    path = "..\\data\\ocr\\"

    # Number of considered examples, for debug reasons. Use "None" to take the whole dataset.
    train_size = 16000
    test_size = 4000

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


def curve(clf, X_train, y_train):
    """Draws the learning curve for the classifier."""

    print("")
    print("*** LEARNING CURVE ***")

    # Create the plot.
    plt.figure()
    plt.title("Learning curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Compute the scores of the learning curve.
    train_sizes, train_scores, val_scores = learning_curve(clf, X_train, y_train, scoring='accuracy', cv=3)

    # Get the mean and standard deviation of train and validation scores.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot the mean for the training scores.
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    # Plot the standard deviation for the training scores.
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")

    # Plot the mean for the validation scores.
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    # Plot the standard deviation for the validation scores.
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")

    # Show the plot.
    plt.ylim(0.05, 1.3)
    plt.legend()
    plt.show()


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
    curve(clf, X_train, y_train, )


# Start the program.
main()