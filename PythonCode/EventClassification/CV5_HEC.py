"""
HierarchicalEventClassifier
@ 2022 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the SVM.
Along with the accuracy, feature importance is calculated.
Two classification is done.
    1) Is it HE or HW?
    2) Is it data from Avoidance trial or Escape trial?
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
from sklearn.model_selection import StratifiedKFold
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
def EventClassifier(matFilePath, numBin, numRepeat=10):
    # Input : matFilePath : Path object

    # Define classification function 
    def fitSVM(X, y, numUnit, numRepeat):
        WholeTestResult = np.zeros([X.shape[0], 3]) # num data x [true, fake, pred]
        PFITestResult = np.zeros([X.shape[0], numUnit, numRepeat])

        # Setup KFold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=622)

        for train_index, test_index in kf.split(X, y):

            X_train = X[train_index, :]
            y_train = y[train_index].copy()
            y_train_shuffle = y[train_index].copy()
            rng.shuffle(y_train_shuffle)

            X_test = X[test_index, :]
            y_test = y[test_index]
            
            clf_real = LinearSVC(penalty='l2', C=0.5, dual=True, max_iter=10000, tol=1e-4)
            clf_real.fit(X_train, y_train)

            clf_fake = LinearSVC(penalty='l2', C=0.5, dual=True, max_iter=10000, tol=1e-4)
            clf_fake.fit(X_train, y_train_shuffle)

            WholeTestResult[test_index,0] = y_test
            WholeTestResult[test_index,1] = clf_fake.predict(X_test)
            WholeTestResult[test_index,2] = clf_real.predict(X_test)

            for unit in range(numUnit):
                for rep in range(numRepeat):
                    X_corrupted = X.copy()
                    # shuffle only selected unit data(=corrupted)
                    for bin in range(numBin):
                        rng.shuffle(X_corrupted[:, numBin*unit + bin])
                    # evaluate corrupted data
                    PFITestResult[test_index, unit, rep] = clf_real.predict(X_corrupted[test_index, :])

        balanced_accuracy = [
                balanced_accuracy_score(WholeTestResult[:,0], WholeTestResult[:,1]),
                balanced_accuracy_score(WholeTestResult[:,0], WholeTestResult[:,2])]

        return [balanced_accuracy, WholeTestResult, PFITestResult]


    # Load Data
    data = loadmat(str(matFilePath.absolute()))
    X = data.get('X')
    y = np.squeeze(data.get('y'))# 1: HE-Avoid, 2: HE-Escape, 3: HW-Avoid, 4: HW-Escape

    numUnit = int(X.shape[1] / numBin)
    print(f'numUnit : {numUnit}')

    # Clip
    X = np.clip(X, -5, 5)

    # Divide datasets
    # 1) HE/HW Classification
    y_HEHW = np.hstack([
            np.ones(np.sum(np.any([(y == 1), (y == 2)], 0)), int) * 1,
            np.ones(np.sum(np.any([(y == 3), (y == 4)], 0)), int) * 2
            ])

    # 2) Avoidance/Escape Classificaion
    X_HE = X[np.any([(y == 1), (y == 2)], 0), :] # remember that the class label starts from 1
    X_HW = X[np.any([(y == 3), (y == 4)], 0), :]
    y_HE = y[np.any([(y == 1), (y == 2)], 0)]
    y_HW = y[np.any([(y == 3), (y == 4)], 0)]

    ########################################################
    #                     Classification                   #
    ########################################################

    # Generate Shuffled Data

    [balanced_accuracy_HEHW, WholeTestResult_HEHW, PFITestResult_HEHW] = fitSVM(X, y_HEHW, numUnit, numRepeat)
    [balanced_accuracy_HEAE, WholeTestResult_HEAE, PFITestResult_HEAE] = fitSVM(X_HE, y_HE, numUnit, numRepeat)
    [balanced_accuracy_HWAE, WholeTestResult_HWAE, PFITestResult_HWAE] = fitSVM(X_HW, y_HW, numUnit, numRepeat)

    return {
        'balanced_accuracy_HEHW' : balanced_accuracy_HEHW,
        'balanced_accuracy_HEAE': balanced_accuracy_HEAE,
        'balanced_accuracy_HWAE': balanced_accuracy_HWAE,
        'WholeTestResult_HEHW': WholeTestResult_HEHW,
        'WholeTestResult_HEAE': WholeTestResult_HEAE,
        'WholeTestResult_HWAE': WholeTestResult_HWAE,
        'PFITestResult_HEHW': PFITestResult_HEHW,
        'PFITestResult_HEAE': PFITestResult_HEAE,
        'PFITestResult_HWAE': PFITestResult_HWAE
        }

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    sessionNames = []

    pbar = tqdm(sorted([p for p in baseFolderPath.glob('#*')]))

    for i, dataPath in enumerate(pbar):
        pbar.set_postfix({'path':dataPath})
        
        sessionName = re.search('(#.*_\wL)', str(dataPath)).groups()[0]

        data_ = EventClassifier(dataPath, 40)
        tankNames.append(str(dataPath))
        sessionNames.append(sessionName)
        result.append(data_)

    return {'tankNames' : tankNames, 'sessionNames': sessionNames, 'result' : result}

if platform.system() == 'Windows':
    output = Batch_EventClassifier(Path(r'D:\Data\Lobster\EventClassificationData_4C'))
    savemat(r'D:/Data/Lobster/CV5_HEC_Result.mat', output)
else:
    output = Batch_EventClassifier(Path(r'/home/ainav/Data/EventClassificationData_4C'))
    savemat(r'/home/ainav/Data/CV5_HEC_Result.mat', output)

