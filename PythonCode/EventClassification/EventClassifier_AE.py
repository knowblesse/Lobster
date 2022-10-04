"""
EventClassifier_AE
@ 2020 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the SVM.
Along with the accuracy, feature importance is calculated.
Divide all dataset into Head Entry and Head Withdrawal, and generate two svm for A/E classification.
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
def EventClassifier(matFilePath, numBin, unit_list):   
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

    # Divide HE and HW dataset

    X_HE = X[np.any([(y == 1), (y == 2)], 0), :]
    X_HW = X[np.any([(y == 3), (y == 4)], 0), :]
    y_HE = y[np.any([(y == 1), (y == 2)], 0)]
    y_HW = y[np.any([(y == 3), (y == 4)], 0)]

    balanced_accuracy_AE = []
    importance_score_AE = []

    # Run Classificaion
    for X, y in zip([X_HE, X_HW], [y_HE, y_HW]):

        # Generate Shuffled Data
        y_real = y.copy()
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)

        # Run Classification 
        y_pred_shuffled = runTest(X, y_shuffled)
        y_pred_real = runTest(X, y_real)

        # Run which unit is important
        numRepeat = 10
        numUnit = int(X.shape[1] / numBin)

        baseScore = balanced_accuracy_score(y_real, y_pred_real)
        importance_score = np.zeros((numRepeat, 2))
        
        # Shuffle units in the list
        for rep in range(numRepeat):
            X_corrupted = X.copy()
            for unit in unit_list-1: # -1 because unit number starts from 1 in the Matlab file
                for bin in range(numBin):
                    rng.shuffle(X_corrupted[:, numBin * unit + bin])
            y_pred_crpt = runTest(X_corrupted, y_real)
            importance_score[rep, 0] = baseScore - balanced_accuracy_score(y_real, y_pred_crpt)

        # Shuffle units NOT in the list
        not_unit_list = list(set(np.arange(numUnit)) - set(unit_list-1))

        for rep in range(numRepeat):
            X_corrupted = X.copy()
            for unit in not_unit_list:
                for bin in range(numBin):
                    rng.shuffle(X_corrupted[:, numBin * unit + bin])
            y_pred_crpt = runTest(X_corrupted, y_real)
            importance_score[rep, 1] = baseScore - balanced_accuracy_score(y_real, y_pred_crpt)

        # Generate output
        balanced_accuracy = [
                balanced_accuracy_score(y_shuffled, y_pred_shuffled),
                balanced_accuracy_score(y_real, y_pred_real)]

        balanced_accuracy_AE.append(balanced_accuracy)
        importance_score_AE.append(importance_score)

    return {
        'balanced_accuracy_HE': balanced_accuracy_AE[0],
        'balanced_accuracy_HW': balanced_accuracy_AE[1],
        'importance_score_HE': importance_score_AE[0],
        'importance_score_HW': importance_score_AE[1],
        }

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    sessionNames = []
    balanced_accuracy = np.empty((0,2))

    pbar = tqdm(sorted([p for p in baseFolderPath.glob('#*')]))

    # unit list of AHW Class 1 unit
    data = loadmat('/home/ainav/Data/EventClassificationData_4C/AHW_C1_unit_list.mat')
    unit_list = data.get('AHW_C1_unit_list')

    for i, dataPath in enumerate(pbar):
        pbar.set_postfix({'path':dataPath})
        
        sessionName = re.search('(#.*_\wL)', str(dataPath)).groups()[0]

        if unit_list[i][0][0][0] != sessionName:
            raise('unit list name does not match with session name')
        if unit_list[i][2][0][0] == 1: # if 0, AHW Class 1 unit is too small or to many
            data_ = EventClassifier(dataPath, 40, unit_list[i][1][0])
            tankNames.append(str(dataPath))
            sessionNames.append(sessionName)
            result.append(data_)

    return {'tankNames' : tankNames, 'result' : result, 'sessionNames': sessionNames}
    
output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData_4C'))
savemat(r'/home/ainav/Data/EventClassificationResult_4C/Output_AE_AHW_C1.mat', output)
