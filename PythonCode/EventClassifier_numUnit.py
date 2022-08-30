"""
EventClassifier_numUnit
@ 2022 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the SVM
This script is a varient of "EventClassifier.py" to test how the number of units affects accuracy.
- Description
    - .mat dataset must have two variable, X and y. (mind the case of the variable name)
    - using the sklearn SVC class, build and test the SVM
    - for the evalutation, Leave One Out method is used
"""
import itertools
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.io import loadmat, savemat
from tqdm import tqdm
import requests
import re

# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

rng = default_rng()
# SVC Event Classifier Function
def EventClassifier_numUnit(matFilePath, numBin, numRepeat):   
    # Input : matFilePath : Path object
    #         numRepeat : num repeat to check the effect of pairs
    # Define Classification function
    def runTest(X,y):
        # Leave One Out, and collect all predict result
        y_pred = np.zeros((len(y),), dtype='uint8')
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = SVC(C=2, kernel='linear')
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred

    # Load Data
    data = loadmat(str(matFilePath.absolute()))
    X = data.get('X')
    y = np.squeeze(data.get('y'))

    # Clip
    X = np.clip(X, -5, 5)

    # Generate Shuffled Data
    y_real = y.copy()
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)

    # Generate array to store accuracies
    numUnit = int(X.shape[1] / numBin)
    balanced_accuracies = np.zeros((2, numRepeat, numUnit)) # D0 : shuffle/real | D1 : repeat | D2 : unitCount

    # Create X_part


    for numUnit2Use in range(1, numUnit+1):
        itercomb = [list(pairs) for pairs in itertools.combinations(np.arange(numUnit), numUnit2Use)] # combinations
        rng.shuffle(itercomb)
        for rep in range(numRepeat): # use only few of the combinations
            X_part = np.empty((X.shape[0],0))
            for unit in itercomb[rep]:
                X_part = np.hstack((X_part, X[:,numBin * unit : numBin * (unit + 1)]))

            # Run Classification
            y_pred_shuffled = runTest(X_part, y_shuffled)
            y_pred_real = runTest(X_part, y_real)

            balanced_accuracies[0, rep, numUnit2Use] = balanced_accuracy_score(y_shuffled, y_pred_shuffled)
            balanced_accuracies[1, rep, numUnit2Use] = balanced_accuracy_score(y_real, y_pred_real)
    return balanced_accuracies

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    sessionNames = []

    pbar = tqdm([p for p in baseFolderPath.glob('#*')])

    for dataPath in pbar:
        pbar.set_postfix({'path':dataPath})
        data_ = EventClassifier_numUnit(dataPath, 40, 5)
        tankNames.append(str(dataPath))
        sessionNames.append(re.search('(#2.*)_event', str(dataPath)).groups()[0])
        result.append(data_)

    return {'tankNames' : tankNames, 'result' : result}
    
output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData'))

savemat(r'/home/ainav/Data/EventClassificationData/Output.mat', output)
