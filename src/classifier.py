import numpy as np
from sklearn.model_selection import KFold, cross_val_score, learning_curve, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt


# ---CONFIGURATION PARAMETERS--- #

# Flag for verbose printing.
VERBOSE = True

# Paths of the datasets and the predictions.
DATA_PATH = "..\\data\\spambase\\"
OUT_PATH = "..\\out\\"

# Number of considered examples, for debug reasons. Use "None" to take the whole dataset.
TRAIN_SIZE = None
TEST_SIZE = None


# ---FUNCTIONS--- #

def printv(text):
    """Prints additional information if the VERBOSE flag is active."""

    if VERBOSE:
        print("[", text, "]")


def load_data():
    """Loads the input data and the output targets from the specified path."""

    print("")
    print("*** DATA LOADING ***")

    # Load the input data.
    printv("Loading the input data...")
    X_train = np.genfromtxt(DATA_PATH + "train-data.csv", delimiter=",")[:TRAIN_SIZE]
    X_test = np.genfromtxt(DATA_PATH + "test-data.csv", delimiter=",")[:TEST_SIZE]

    # Load the output targets.
    printv("Loading the output targets...")
    y_train = np.genfromtxt(DATA_PATH + "train-targets.csv", dtype="str")[:TRAIN_SIZE]
    y_test = np.genfromtxt(DATA_PATH + "test-targets.csv", dtype="str")[:TEST_SIZE]

    print("Number of training examples: ", len(X_train))
    print("Number of test examples: ", len(X_test))

    return X_train, X_test, y_train, y_test


def cross_validation(X_train, y_train):
    """Performs k-fold cross-validation to tune the model's parameters."""

    print("")
    print("*** CROSS VALIDATION ***")

    # Define the cross-validation process.
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Specify the candidate values.
    gamma_values = [0.1, 0.05, 0.02, 0.01]
    accuracy_scores = []

    # Perform cross-validation for each candidate value.
    printv("Starting the cross-validation...")
    for gamma in gamma_values:
        # Train the classifier.
        clf = SVC(C=10, kernel='rbf', gamma=gamma)

        # Compute the cross-validation scores.
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


def gs_cross_validation(svc, X_train, y_train):
    """Performs grid search cross-validation to optimize two parameters at the same time."""

    print("")
    print("*** GRID SEARCH CROSS-VALIDATION ***")

    # Candidate values for the parameters C and gamma.
    possible_parameters = {
        'C': [1e0, 1e1, 1e2, 1e3],
        'gamma': [1e-1, 1e-2, 1e-3, 1e-4]
    }

    # Obtain the classifier directly from the grid search cross-validation.
    printv("Starting the grid search cross-validation...")
    clf = GridSearchCV(svc, possible_parameters, n_jobs=4, cv=3, iid=False)

    return clf


def train(clf, X_train, y_train):
    """Fits the classifier to the training set."""

    print("")
    print("*** TRAINING ***")

    # Initialize and train the classifier.
    printv("Training the classifier...")
    clf.fit(X_train, y_train)

    return clf


def test(clf, X_test, y_test):
    """Evaluates the classifier on the test set and returns the predictions."""

    print("")
    print("*** TESTING ***")

    # Predict the test targets with the classifier.
    printv("Predicting the test targets...")
    y_pred = clf.predict(X_test)

    # Compute the test accuracy.
    printv("Computing the accuracy...")
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Generate the classification report.
    printv("Generating the classification report...")
    report = metrics.classification_report(y_test, y_pred, output_dict=False)
    print(report)

    print("The test accuracy is ", accuracy)

    return y_pred


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
    printv("Computing the scores through cross-validation....")
    train_sizes, train_scores, val_scores = learning_curve(clf, X_train, y_train, scoring='accuracy', cv=3)

    # Get the mean and standard deviation of train and validation scores.
    printv("Computing the mean and standard deviation of the scores...")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot the mean for the training scores.
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    # Plot the standard deviation for the training scores.
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")

    # Plot the mean for the validation scores.
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    # Plot the standard deviation for the validation scores.
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")

    # Show the plot.
    plt.ylim(0.05, 1.3)
    plt.legend()
    plt.show()


# ---MAIN--- #

def main():
    """Main function."""

    GS = True

    # Load the data.
    X_train, X_test, y_train, y_test = load_data()

    # Obtain the optimal parameters through grid search cross-validation.
    svc = SVC(kernel='rbf')
    clf = gs_cross_validation(svc, X_train, y_train)

    # Train the optimal classifier.
    clf = train(clf, X_train, y_train)

    # Test the classifier and obtain the predictions.
    y_pred = test(clf, X_test, y_test)

    # Draw the learning curve.
    curve(clf, X_train, y_train)

    # Save the predictions to file.
    np.savetxt(OUT_PATH + "test-pred.txt", y_pred, fmt='%s', delimiter='\n')


# Start the program.
main()
