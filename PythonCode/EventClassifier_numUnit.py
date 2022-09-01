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

def generateNonRepeatedCombination(numIndex, setSize, repeat):
    elements = [i for i in range(numIndex)]
    selected_index = set()
    rng = default_rng()
    while len(selected_index) < repeat:
        permuted_index = rng.permutation([i for i in range(len(elements))])
        candidate = tuple(sorted(permuted_index[0:setSize]))
        if not (candidate in selected_index):
            selected_index.add(candidate)
    return list(selected_index)




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

    pbar1 = tqdm(np.arange(1, numUnit))
    for numUnit2Use in pbar1:
        pbar1.set_postfix({'numUnit2Use': numUnit2Use, 'numUnit': numUnit})
        itercomb = generateNonRepeatedCombination(numUnit, numUnit2Use, numRepeat)
        rng.shuffle(itercomb)
        for rep in range(numRepeat): # use only few of the combinations
            print(f'numUnit : {numUnit2Use} rep : {rep}')
            X_part = np.empty((X.shape[0],0))
            for unit in itercomb[rep]:
                X_part = np.hstack((X_part, X[:,numBin * unit : numBin * (unit + 1)]))

            # Run Classification
            y_pred_shuffled = runTest(X_part, y_shuffled)
            y_pred_real = runTest(X_part, y_real)

            balanced_accuracies[0, rep, numUnit2Use-1] = balanced_accuracy_score(y_shuffled, y_pred_shuffled)
            balanced_accuracies[1, rep, numUnit2Use-1] = balanced_accuracy_score(y_real, y_pred_real)
    # Run Classification with full model
    y_pred_shuffled = runTest(X, y_shuffled)
    y_pred_real = runTest(X, y_real)

    balanced_accuracies[0, :, -1] = balanced_accuracy_score(y_shuffled, y_pred_shuffled)
            
    return balanced_accuracies

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    sessionNames = []

    pbar = tqdm([p for p in baseFolderPath.glob('#*')])

    for dataPath in pbar:
        pbar.set_postfix({'path':dataPath})
        tankNames.append(str(dataPath))
        sessionNames.append(re.search('(#2.*)_event', str(dataPath)).groups()[0])
        data_ = EventClassifier_numUnit(dataPath, 40, 5)
        result.append(data_)

    return {'tankNames' : tankNames, 'result' : result}

output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData'))
savemat(r'/home/ainav/Data/EventClassificationData/Output.mat', output)

