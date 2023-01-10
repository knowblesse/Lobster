"""
HierarchicalEventClassifier
@ 2022 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the Bernoulli Naive Bayes Classifier
Along with the accuracy, feature importance is calculated.
Two classification is done.
    1) Is it HE or HW?
    2) Is it data from Avoidance trial or Escape trial?
- Description
    - .mat dataset must have two variable, X and y. (mind the case of the variable name)
    - using the sklearn BernulliNB class, build and test the classifier
    - for the evaluation, 5-fold Cross Validation method is used
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
        CrossEntropy = 0
        PFITestResult = np.zeros([X.shape[0], numUnit, numRepeat])
        PFICrossEntropy = np.zeros([numUnit,1])
        FeatureProb = np.zeros([CV_split, 2, X.shape[1]])

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
            
            # Permutation Feature Importance
            for unit in range(numUnit):
                for rep in range(numRepeat):
                    X_corrupted = X.copy()
                    # shuffle only selected unit data(=corrupted)
                    # for bin in range(numBin):
                    #     rng.shuffle(X_corrupted[:, numBin*unit + bin])
                    # # evaluate corrupted data
                    # PFITestResult[test_index, unit, rep] = clf_real.predict_proba(X_corrupted[test_index, :])[:,1]
                    rng.shuffle(X_corrupted[:, numBin*unit : numBin*(unit+1)])
                    # evaluate corrupted data
                    PFITestResult[test_index, unit, rep] = clf_real.predict_proba(X_corrupted[test_index, :])[:,1]

            # Feature Probability
            FeatureProb[cv_index, :, :] = clf_real.feature_log_prob_
            clf_real.predict_proba(X_test[1:2, :])

        # Calculate Cross Entropy
        CrossEntropy = log_loss(WholeTestResult[:,0], WholeTestResult[:,2])

        for unit in range(numUnit):
            crossEntropy_ = np.zeros([numRepeat])
            for rep in range(numRepeat):
                crossEntropy_[rep] = log_loss(WholeTestResult[:,0], PFITestResult[:, unit, rep])
            PFICrossEntropy[unit] = np.mean(crossEntropy_) - CrossEntropy

        balanced_accuracy = [
            balanced_accuracy_score(WholeTestResult[:,0], WholeTestResult[:,1] >= 0.5),
            balanced_accuracy_score(WholeTestResult[:,0], WholeTestResult[:,2] >= 0.5)]

        return [balanced_accuracy, CrossEntropy, WholeTestResult, PFITestResult, PFICrossEntropy, FeatureProb]


    # Load Data
    data = loadmat(str(matFilePath.absolute()))
    X = data.get('X')
    y = np.squeeze(data.get('y'))# 1: HE-Avoid, 2: HE-Escape, 3: HW-Avoid, 4: HW-Escape

    numUnit = int(X.shape[1] / numBin)

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

    [balanced_accuracy_HEHW, CrossEntropy_HEHW, WholeTestResult_HEHW, PFITestResult_HEHW, PFICrossEntropy_HEHW, FeatureProb_HEHW] = fitBNB(X, y_HEHW, numUnit, numRepeat)
    [balanced_accuracy_HEAE, CrossEntropy_HEAE, WholeTestResult_HEAE, PFITestResult_HEAE, PFICrossEntropy_HEAE, FeatureProb_HEAE] = fitBNB(X_HE, y_HE, numUnit, numRepeat)
    [balanced_accuracy_HWAE, CrossEntropy_HWAE, WholeTestResult_HWAE, PFITestResult_HWAE, PFICrossEntropy_HWAE, FeatureProb_HWAE] = fitBNB(X_HW, y_HW, numUnit, numRepeat)

    return {
        'balanced_accuracy_HEHW' : balanced_accuracy_HEHW,
        'balanced_accuracy_HEAE': balanced_accuracy_HEAE,
        'balanced_accuracy_HWAE': balanced_accuracy_HWAE,
        'CrossEntropy_HEHW': CrossEntropy_HEHW,
        'CrossEntropy_HEAE': CrossEntropy_HEAE,
        'CrossEntropy_HWAE': CrossEntropy_HWAE,
        'WholeTestResult_HEHW': WholeTestResult_HEHW,
        'WholeTestResult_HEAE': WholeTestResult_HEAE,
        'WholeTestResult_HWAE': WholeTestResult_HWAE,
        'PFITestResult_HEHW': PFITestResult_HEHW,
        'PFITestResult_HEAE': PFITestResult_HEAE,
        'PFITestResult_HWAE': PFITestResult_HWAE,
        'PFICrossEntropy_HEHW': PFICrossEntropy_HEHW,
        'PFICrossEntropy_HEAE': PFICrossEntropy_HEAE,
        'PFICrossEntropy_HWAE': PFICrossEntropy_HWAE,
        'feature_prob_HEHW': FeatureProb_HEHW,
        'feature_prob_HEAE': FeatureProb_HEAE,
        'feature_prob_HWAE': FeatureProb_HWAE,
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
        balancedAccuracy = np.vstack([balancedAccuracy, np.array([data_['balanced_accuracy_HEAE'][1], data_['balanced_accuracy_HWAE'][1]])])
    print(np.mean(balancedAccuracy, 0))
    return {'tankNames' : tankNames, 'sessionNames': sessionNames, 'result' : result}


# output = Batch_EventClassifier(Path(r'D:\Data\Lobster\EventClassificationData_4C_[-1200,-200]'))
# savemat(r'D:/Data/Lobster/BNB_Result_unitshffle_[-1200,-200].mat', output)

outputData = np.zeros([40,9])
for i in np.arange(1, 10):
    output = Batch_EventClassifier(Path(r'D:\Data\Lobster\EventClassificationData_4C_'+str(i)))
    for j in np.arange(40):
        outputData[j, i-1] = output['result'][j]['balanced_accuracy_HWAE'][1]
