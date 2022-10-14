"""
EventClassifier_HEHW
@ 2020 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the SVM.
Along with the accuracy, feature importance is calculated.
Only classifiy whether the data is from HE or HW
- Description
    - .mat dataset must have two variable, X and y. (mind the case of the variable name)
    - using the sklearn SVC class, build and test the SVM
    - for the evalutation, Leave One Out method is used
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
from tqdm import tqdm
import requests
import re
# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

rng = default_rng()
# SVC Event Classifier Function
def EventClassifier(matFilePath, numBin):   
    # Input : matFilePath : Path object
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

    # Remove A/E Information
    y = np.hstack([
            np.ones(np.sum(np.any([(y == 1), (y == 2)], 0)), int) * 1,
            np.ones(np.sum(np.any([(y == 3), (y == 4)], 0)), int) * 2
            ])

    # Run Classificaion

    # Generate Shuffled Data
    y_real = y.copy()
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)

    # Run Classification 
    y_pred_shuffled = runTest(X, y_shuffled)
    y_pred_real = runTest(X, y_real)

    # Run which unit is important
    numRepeat = 30
    numUnit = int(X.shape[1] / numBin)

    baseScore = balanced_accuracy_score(y_real, y_pred_real)
    importance_score = np.zeros((numRepeat, numUnit))

    for unit in range(numUnit):
        for rep in range(numRepeat):
            X_corrupted = X.copy()
            for bin in range(numBin):
                rng.shuffle(X_corrupted[:, numBin * unit + bin])
            y_pred_crpt = runTest(X_corrupted, y_real)
            importance_score[rep, unit] = baseScore - balanced_accuracy_score(y_real, y_pred_crpt)

    # Generate output
    balanced_accuracy = [
            balanced_accuracy_score(y_shuffled, y_pred_shuffled),
            balanced_accuracy_score(y_real, y_pred_real)]

    return {
        'balanced_accuracy': balanced_accuracy,
        'importance_score': importance_score,
        }

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    sessionNames = []
    balanced_accuracy = np.empty((0,2))

    pbar = tqdm(sorted([p for p in baseFolderPath.glob('#*')]))

    for dataPath in pbar:
        pbar.set_postfix({'path':dataPath})
        data_ = EventClassifier(dataPath, 40)
        tankNames.append(str(dataPath))
        sessionNames.append(re.search('(#.*_\wL)', str(dataPath)).groups()[0])
        result.append(data_)

    return {'tankNames' : tankNames, 'result' : result, 'sessionNames': sessionNames}
    
output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData_4C'))
savemat(r'/home/ainav/Data/EventClassificationResult_4C/Output_HEHW.mat', output)
