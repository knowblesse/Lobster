"""
EventClassifier
@ 2020 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the SVM.
Along with the accuracy, feature importance is calculated.
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

    importance_score = np.zeros((numRepeat, numUnit))

    # importance to the specific class
    importance_score_AHE = np.zeros((numRepeat, numUnit))
    importance_score_EHE = np.zeros((numRepeat, numUnit))
    importance_score_AHW = np.zeros((numRepeat, numUnit))
    importance_score_EHW = np.zeros((numRepeat, numUnit))

    baseScore = balanced_accuracy_score(y_real, y_pred_real)
    baseConfusion = confusion_matrix(y_real, y_pred_real)
    baseScore_AHE = baseConfusion[0, 0] / np.sum(y_real == 1) # matlab class starts from class 1
    baseScore_EHE = baseConfusion[1, 1] / np.sum(y_real == 2)
    baseScore_AHW = baseConfusion[2, 2] / np.sum(y_real == 3)
    baseScore_EHW = baseConfusion[3, 3] / np.sum(y_real == 4)
    for unit in range(numUnit):
        for rep in range(numRepeat):
            X_corrupted = X.copy()
            for bin in range(numBin):
                rng.shuffle(X_corrupted[:, numBin * unit + bin])
            y_pred_crpt = runTest(X_corrupted, y_real)
            importance_score[rep, unit] = baseScore - balanced_accuracy_score(y_real, y_pred_crpt)
            conf_ = confusion_matrix(y_real, y_pred_crpt) 
            
            importance_score_AHE[rep, unit] = baseScore_AHE - (conf_[0,0] / np.sum(y_real == 1))
            importance_score_EHE[rep, unit] = baseScore_EHE - (conf_[1,1] / np.sum(y_real == 2))
            importance_score_AHW[rep, unit] = baseScore_AHW - (conf_[2,2] / np.sum(y_real == 3))
            importance_score_EHW[rep, unit] = baseScore_EHW - (conf_[3,3] / np.sum(y_real == 4)) 

    # Generate output
    accuracy = [
            accuracy_score(y_shuffled, y_pred_shuffled),
            accuracy_score(y_real, y_pred_real)]
    balanced_accuracy = [
            balanced_accuracy_score(y_shuffled, y_pred_shuffled),
            balanced_accuracy_score(y_real, y_pred_real)]
    conf_matrix = [
            confusion_matrix(y_shuffled, y_pred_shuffled),
            confusion_matrix(y_real, y_pred_real)]
    return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'confusion_matrix': conf_matrix,
            'importance_score': importance_score,
            'importance_score_AHE': importance_score_AHE,
            'importance_score_EHE': importance_score_EHE,
            'importance_score_AHW': importance_score_AHW,
            'importance_score_EHW': importance_score_EHW
            }

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    importance_score = np.empty((0,2))
    balanced_accuracy = np.empty((0,2))

    pbar = tqdm([p for p in baseFolderPath.glob('#*')])

    for dataPath in pbar:
        pbar.set_postfix({'path':dataPath})
        data_ = EventClassifier(dataPath, 40)
        tankNames.append(str(dataPath))
        result.append(data_)
        balanced_accuracy = np.vstack([
            balanced_accuracy,
            np.expand_dims(np.array(data_['balanced_accuracy']),0)])

    return {'tankNames' : tankNames, 'result' : result, 'balanced_accuracy' : balanced_accuracy}
    
output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData_4C'))
print(f'shuffled : {np.mean(output["balanced_accuracy"],0)[0]:.2f} ±{np.std(output["balanced_accuracy"],0)[0]:.2f}')
print(f'    real : {np.mean(output["balanced_accuracy"],0)[1]:.2f} ±{np.std(output["balanced_accuracy"],0)[1]:.2f}')
savemat(r'/home/ainav/Data/EventClassificationResult_4C/Output.mat', output)
