"""
EventClassifier_numUnit
@ 2022 Knowblesse
Modified Version of EventClassifier. This script output accuracy when only a subset of units are used in the
classification.
- Description
    - .mat dataset must have two variable, X and y. (mind the case of the variable name)
    - using the sklearn SVC class, build and test the SVM
    - for the evalutation, Leave One Out method is used
"""
from multiprocessing import Pool
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat, savemat
import re
import os
import requests

# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

def generateNonRepeatedCombination(numIndex, setSize, repeat):
    # Pick a subset size of (setSize) from range(numIndex) for (repeat) times
    elements = [i for i in range(numIndex)]
    selected_index = set()
    rng = default_rng()
    while len(selected_index) < repeat:
        permuted_index = rng.permutation([i for i in range(len(elements))])
        candidate = tuple(sorted(permuted_index[0:setSize]))
        if not (candidate in selected_index):
            selected_index.add(candidate)
    return list(selected_index)

def runTest(X,y):
    # Leave One Out, and collect all predict result
    y_pred = np.zeros((len(y),), dtype='uint8')
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = SVC(C=2, kernel='linear', max_iter=100000)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def runTestWithPartX(X, comb, numBin, y_shuffled, y_real):
    X_part = np.empty((X.shape[0], 0))
    for unit in comb:
        X_part = np.hstack((X_part, X[:, numBin * unit: numBin * (unit + 1)]))

    # Run Classification
    y_pred_shuffled = runTest(X_part, y_shuffled)
    y_pred_real = runTest(X_part, y_real)

    return [balanced_accuracy_score(y_shuffled, y_pred_shuffled), balanced_accuracy_score(y_real, y_pred_real)]

if __name__ == '__main__':
    rng = default_rng()

    numBin = 40
    numRepeat = 5

    dataList = sorted([p for p in Path(r'E:\EventClassificationData').glob('#*')]) # /home/ainav/Data/EventClassificationData
    isPass = False

    for dataPath in dataList:
        tankNames = str(dataPath)
        sessionNames = re.search('(#2.*)_event', str(dataPath)).groups()[0]

        # if sessionNames[-1] == '#21AUG4-211027-184953_PL':
        #     isPass = True
        # if not isPass:
        #     continue

        print('\n')
        print(sessionNames)

        # SVC Event Classifier Function
        # Load Data
        data = loadmat(str(dataPath.absolute()))
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

        print(f'Num Unit : {numUnit}')
        for numUnit2Use in np.arange(1, numUnit):
            print(f'{numUnit2Use}-', end='')
            itercomb = generateNonRepeatedCombination(numUnit, numUnit2Use, numRepeat)
            rng.shuffle(itercomb)
            params = []
            for i in range(numRepeat):
                params.append([X, itercomb[i], numBin, y_shuffled, y_real])
            with Pool(5) as p:
                result = p.starmap(runTestWithPartX, params)
            for i in range(numRepeat):
                balanced_accuracies[0, i, numUnit2Use - 1] = result[i][0]
                balanced_accuracies[1, i, numUnit2Use - 1] = result[i][1]

        # Run Classification with full model
        y_pred_shuffled = runTest(X, y_shuffled)
        y_pred_real = runTest(X, y_real)

        balanced_accuracies[0, :, -1] = balanced_accuracy_score(y_shuffled, y_pred_shuffled)
        balanced_accuracies[1, :, -1] = balanced_accuracy_score(y_real, y_pred_real)

        result = {'result': balanced_accuracies}
        savemat(r'E:\EventClassificationResult\\'+sessionNames+'_Output.mat', result)
        requests.get(f'https://api.telegram.org/bot5269105245:AAHyOGSwCeYvmNNT3nKzQ6Ho_KfVw2nKTYE/sendMessage?chat_id=5520161508&text={sessionNames}')

