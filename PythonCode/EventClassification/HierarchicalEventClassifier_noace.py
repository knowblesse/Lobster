"""
HierarchicalEventClassifier_noace
@ 2024 Knowblesse
Similar to `HierarchicalEventClassifier`.
But in this script, I manually remove some neurons from the dataset to test the predictiveness of the model.
"""
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelBinarizer
from scipy.io import loadmat, savemat
from tqdm import tqdm
import re
import platform
import argparse

# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

rng = default_rng()

# SVC Event Classifier Function
def EventClassifier(matFilePath, numBin, numRepeat=10):
    # Input : matFilePath : Path object

    # Define classification function 
    def fitBNB(X, y, numUnit, numRepeat):
        CV_split = 5

        # Init. array to hold output info
        WholeTestResult = np.zeros([X.shape[0], 3]) # num data x [true, fake, pred]

        # Binarize y
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
        y = np.ravel(y)

        # Setup KFold
        kf = StratifiedKFold(n_splits=CV_split, shuffle=True, random_state=622)

        for cv_index, [train_index, test_index] in enumerate(kf.split(X, y)):

            X_train = X[train_index, :]
            y_train = y[train_index].copy()
            y_train_shuffle = y[train_index].copy()
            rng.shuffle(y_train_shuffle)

            X_test = X[test_index, :]
            y_test = y[test_index]
            
            clf_real = BernoulliNB(fit_prior=False)
            clf_real.fit(X_train, y_train)

            clf_fake = BernoulliNB(fit_prior=False)
            clf_fake.fit(X_train, y_train_shuffle)

            WholeTestResult[test_index,0] = y_test
            WholeTestResult[test_index,1] = clf_fake.predict_proba(X_test)[:,1] # prob. of being 1
            WholeTestResult[test_index,2] = clf_real.predict_proba(X_test)[:,1] # prob. of being 1
            
        balanced_accuracy = [
            balanced_accuracy_score(WholeTestResult[:,0], WholeTestResult[:,1] >= 0.5),
            balanced_accuracy_score(WholeTestResult[:,0], WholeTestResult[:,2] >= 0.5)]

        return [balanced_accuracy, WholeTestResult]


    # Load Data
    data = loadmat(str(matFilePath.absolute()))
    X = data.get('X')
    y = np.squeeze(data.get('y'))# 1: HE-Avoid, 2: HE-Escape, 3: HW-Avoid, 4: HW-Escape

    numUnit = int(X.shape[1] / numBin)

    # Remove ace neurons
    FI_rank = data.get('seq')
    num_ace2remove = int(np.round(numUnit * 0.2))

    units_to_use = FI_rank[:-num_ace2remove]

    X_noace = np.zeros([X.shape[0], 0])

    for unit in units_to_use:
        target_unit = unit[0] - 1 # 1-based index to 0-based index
        X_noace = np.hstack([X_noace, X[:, target_unit*numBin:(target_unit+1)*numBin]])

    print(X_noace.shape)

    # Clip
    X = np.clip(X_noace, -5, 5)

    # Avoidance/Escape Classificaion
    X_HW = X[np.any([(y == 3), (y == 4)], 0), :]
    y_HW = y[np.any([(y == 3), (y == 4)], 0)]

    ########################################################
    #                     Classification                   #
    ########################################################

    # Generate Shuffled Data

    [balanced_accuracy_HWAE, WholeTestResult_HWAE] = fitBNB(X_HW, y_HW, numUnit, numRepeat)

    return {
        'balanced_accuracy_HWAE': balanced_accuracy_HWAE,
        'WholeTestResult_HWAE': WholeTestResult_HWAE
        }

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    sessionNames = []
    balancedAccuracy = np.zeros([0,2])

    pbar = tqdm(sorted([p for p in baseFolderPath.glob('#*')]))

    for i, dataPath in enumerate(pbar):
        pbar.set_postfix({'path':dataPath})
        
        sessionName = re.search('(#.*_\wL)', str(dataPath)).groups()[0]

        data_ = EventClassifier(dataPath, 40)
        tankNames.append(str(dataPath))
        sessionNames.append(sessionName)
        result.append(data_)
    return {'tankNames' : tankNames, 'sessionNames': sessionNames, 'result' : result}

def Batch_Batch_EventClassifier(basebaseFolderPath):
    # batch of the "Batch_EventClassifier" function.
    # for predictiveness testing using multiple neural datasets from different timewindow

    for basePath in basebaseFolderPath.glob('*'):
        print(f'running {basePath.stem}')
        output = Batch_EventClassifier(basePath)
        savemat(str((basebaseFolderPath.absolute().parent / (basePath.stem + '_NonOverlap.mat')).absolute()), output)

parser = argparse.ArgumentParser(prog='HierarchicalEventClassifier')
parser.add_argument('predictive')
args = parser.parse_args()

if platform.system() == 'Windows':
    baseFolder = Path(r'D:\Data\Lobster')
else:
    baseFolder = Path(r'/home/ainav/Data')

if args.predictive == 'true':
    print('running predictive')
    Batch_Batch_EventClassifier(baseFolder / 'EventClassificationData_4C_Predictive_NonOverlap')
elif args.predictive == 'false':
    print('running on single dataset')
    output = Batch_EventClassifier(baseFolder / 'EventClassificationData_4C')
    savemat(str((baseFolder / 'BNB_Result_unitshffle_noace.mat').absolute()), output)

