"""
EventClassifier_AE
@ 2022 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the SVM.
Along with the accuracy, feature importance is calculated.
Divide all dataset into Head Entry and Head Withdrawal, and generate two svm for A/E classification.
- Description
    - .mat dataset must have two variable, X and y. (mind the case of the variable name)
    - using the sklearn SVC class, build and test the SVM
    - for the evaluation, Leave One Out method is used
"""
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat, savemat
from tqdm import tqdm
import re
import platform

# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

rng = default_rng()

# SVC Event Classifier Function
def EventClassifier(matFilePath, numBin):
    # Input : matFilePath : Path object
    # Define Classification function
    def fitSVM(X,y):
        # Leave One Out, and collect all predict result
        y_pred = np.zeros((len(y),), dtype=int)
        weights = np.zeros((len(y), X.shape[1]))
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = LinearSVC(penalty='l2', C=0.5, dual=True, max_iter=10000, tol=1e-4)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
            weights[test_index, :] = np.abs(clf.coef_)
        return [y_pred, np.mean(weights, axis=0)]

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

    balanced_accuracy_AE = []
    unitRank_AE = []
    accuracy_AE = []
    importanceScore_AE = []
    importanceUnit_AE = []

    # Run Classificaion
    for X, y in zip([X_HE, X_HW], [y_HE, y_HW]):

        # Generate Shuffled Data
        y_real = y.copy()
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)

        # Run Control Classification
        y_pred_shuffled, _ = fitSVM(X, y_shuffled)

        # Recursive Feature Elimination
        unitList = np.arange(int(X.shape[1] / numBin))
        unitRank = []
        accuracy = []
        while len(unitList) > 0:
            # Generate partial X
            X_partial = np.empty((X.shape[0], 0))
            for i in unitList:
                X_partial = np.hstack((X_partial, X[:, i * numBin: (i + 1) * numBin]))
            y_pred_partial, weights = fitSVM(X_partial, y_real)
            accuracy.append(balanced_accuracy_score(y_real, y_pred_partial))
            # Find the least important unit
            leastImportantUnitIndex = unitList[np.argmin(np.max(np.reshape(weights, (numBin, -1), order='F'), 0))]
            unitRank.append(leastImportantUnitIndex)
            unitList = np.delete(unitList, np.where(unitList == leastImportantUnitIndex))

        # Permutation Feature importance
        # - first, take only the feature at the highest accuracy, and then do the PFI calculation
        max_accuracy = np.max(accuracy)
        max_accuracy_index = np.argmax(accuracy)

        unitList = unitRank[max_accuracy_index:]
        numRepeat = 30
        
        importanceScore = np.zeros((numRepeat, len(unitList)))
        for unitIndex, unit in enumerate(sorted(unitList)):
            for rep in range(numRepeat):
                X_corrupted = X.copy()
                for bin in range(numBin):
                    rng.shuffle(X_corrupted[:, numBin * unit + bin])
                y_pred_corrupted = fitSVM(X_corrupted, y_real)
                importanceScore[rep,unitIndex] = max_accuracy - balanced_accuracy_score(y_real, y_pred_corrupted)

        importanceScore = np.mean(importanceScore, 0)

        balanced_accuracy = [
            balanced_accuracy_score(y_shuffled, y_pred_shuffled),
            accuracy[0],
            max_accuracy]

        balanced_accuracy_AE.append(balanced_accuracy)
        unitRank_AE.append(unitRank)
        accuracy_AE.append(accuracy)
        importanceScore_AE.append(importanceScore)
        importanceUnit_AE.append(sorted(unitList))

    return {
        'balanced_accuracy_HE': balanced_accuracy_AE[0],
        'balanced_accuracy_HW': balanced_accuracy_AE[1],
        'unitRank_HE' : unitRank_AE[0],
        'unitRank_HW': unitRank_AE[1],
        'accuracy_HE' : accuracy_AE[0],
        'accuracy_HW': accuracy_AE[1],
        'importanceScore_HE' : importanceScore_AE[0],
        'importanceScore_HW' : importanceScore_AE[1],
        'importanceUnit_HE' : importanceUnit_AE[0],
        'importanceUnit_HW' : importanceUnit_AE[1],
        }

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    sessionNames = []
    balanced_accuracy = np.empty((0,2))

    pbar = tqdm(sorted([p for p in baseFolderPath.glob('#*')]))

    for i, dataPath in enumerate(pbar):
        pbar.set_postfix({'path':dataPath})
        
        sessionName = re.search('(#.*_\wL)', str(dataPath)).groups()[0]

        data_ = EventClassifier(dataPath, 40)
        tankNames.append(str(dataPath))
        sessionNames.append(sessionName)
        result.append(data_)
        balanced_accuracy = np.vstack((balanced_accuracy, data_['balanced_accuracy_HW']))

    print(f"{np.mean(balanced_accuracy, 0)[0]} | {np.mean(balanced_accuracy, 0)[1]}")
    return {'tankNames' : tankNames, 'result' : result, 'sessionNames': sessionNames}

if platform.system() == 'Windows':
    output = Batch_EventClassifier(Path(r'D:\Data\Lobster\EventClassificationData_4C'))
    savemat(r'D:\Data\Lobster\Output_AE_RFE.mat', output)
else:
    output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData_4C'))
    savemat(r'/home/ainav/Data/EventClassificationResult_4C/Output_AE_RFE_max_FI.mat', output)
