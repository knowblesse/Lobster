"""
EC_GridSearch
@ 2022 Knowblesse
Find the best hyperparameter for EV
"""
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.io import loadmat, savemat
import platform

# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

rng = default_rng()
# SVC Event Classifier Function
def EventClassifier(matFilePath, numBin, kernel, C, gamma=1, coef0=0, degree=3):
    # Input : matFilePath : Path object
    # Define Classification function
    def runTest(X, y, kernel, C, gamma, coef0, degree):
        # Leave One Out, and collect all predict result
        y_pred = np.zeros((len(y),), dtype='uint8')
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = SVC(C=C, kernel=kernel, coef0=coef0, gamma=gamma, degree=degree)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred

    # Load Data
    data = loadmat(str(matFilePath.absolute()))
    X = data.get('X')
    y = np.squeeze(data.get('y'))

    # Clip
    X = np.clip(X, -5, 5)

    # Divide HE and HW dataset
    X_HE = X[np.any([(y == 1), (y == 2)], 0), :] # remember that the class label starts from 1
    X_HW = X[np.any([(y == 3), (y == 4)], 0), :]
    y_HE = y[np.any([(y == 1), (y == 2)], 0)]
    y_HW = y[np.any([(y == 3), (y == 4)], 0)]

    # Only focus on the discrimination of A/E on HW dataset
    X = X_HW
    y = y_HW

    # Generate Shuffled Data
    y_real = y.copy()
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)

    # Run Classification 
    y_pred_shuffled = runTest(X, y_shuffled, kernel, C, gamma, coef0, degree)
    y_pred_real = runTest(X, y_real, kernel, C, gamma, coef0, degree)

    return [
            balanced_accuracy_score(y_shuffled, y_pred_shuffled),
            balanced_accuracy_score(y_real, y_pred_real)]

def printResult(baseFolderPath, kernel, C, gamma, coef0, degree):
    # run through all dataset
    result = np.empty((0,2))
    for dataPath in sorted([p for p in baseFolderPath.glob('#*')]):
        result = np.vstack((result, EventClassifier(dataPath, 40, kernel, C, gamma, coef0, degree)))
    mean_result = np.mean(result, 0)
    print(f"{mean_result[0]:6.2f} | {mean_result[1]:6.2f} | kernel:{kernel:s} | C:{C:.1f} | gamma:{gamma} | coef0:{coef0} | degree:{degree}")


def Batch_EventClassifier(baseFolderPath):
    for C in [2, 4, 6, 8, 10, 12]:
        for coef0 in np.arange(-1, 1, 0.1):
            printResult(baseFolderPath, 'sigmoid', C, gamma='scale', coef0=coef0, degree=3)

if platform.system() == 'Windows':
    output = Batch_EventClassifier(Path(r'D:\Data\Lobster\EventClassificationData_4C'))
else:
    output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData_4C'))
