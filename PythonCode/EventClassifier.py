"""
EventClassifier
@ 2020 Knowblesse
Using the preprocessed Neural Ensemble dataset with behavior labels, build and test the SVM
- Description
    - .mat dataset must have two variable, X and y. (mind the case of the variable name)
    - using the sklearn SVC class, build and test the SVM
    - for the evalutation, Leave One Out method is used
"""

import numpy as np
from pathlib import Path
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.io import loadmat, savemat
from tqdm import tqdm
from sklearn.inspection import permutation_importance

# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

# SVC Event Classifier Function
def EventClassifier(matFilePath):   
    # Input : matFilePath : Path object
    # Define Classification function
    def runTest(X,Y):
        # Leave One Out, and collect all predict result
        Y_pred = np.zeros((len(Y),), dtype='uint8')
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            clf = SVC(C=2, kernel='linear')
            clf.fit(X_train, Y_train)
            Y_pred[test_index] = clf.predict(X_test)
        return Y_pred

    def getCoefImportance(X, Y, n_timebin = 20):
        clf = SVC(C=2, kernel='linear')
        clf.fit(X, Y)
        mean_coef = np.mean((clf.coef_ ** 2), 0)
        return np.sum(np.reshape(mean_coef, (n_timebin, -1)), 0)

    # Load Data
    data = loadmat(str(matFilePath.absolute()))
    print(str(matFilePath) + ' is loaded \n')
    X = data.get('X')
    Y = data.get('y')
    Y = np.squeeze(Y)

    # Clip
    X = np.clip(X, -5, 5)

    # Generate Shuffled Data
    Y_real = Y.copy()
    Y_shuffled = Y.copy()
    np.random.shuffle(Y_shuffled)

    # Run Classification 
    Y_pred_shuffled = runTest(X, Y_shuffled)
    Y_pred_real = runTest(X,Y_real)
    importance_score = getCoefImportance(X, Y_real)

    # Generate output
    accuracy = [
            accuracy_score(Y_shuffled, Y_pred_shuffled),
            accuracy_score(Y_real, Y_pred_real)]
    balanced_accuracy = [
            balanced_accuracy_score(Y_shuffled, Y_pred_shuffled),
            balanced_accuracy_score(Y_real, Y_pred_real)]
    conf_matrix = [
            confusion_matrix(Y_shuffled, Y_pred_shuffled),
            confusion_matrix(Y_real, Y_pred_real)]
    return {
            'accuracy' : accuracy, 
            'balanced_accuracy' : balanced_accuracy,
            'confusion_matrix' : conf_matrix,
            'importance_score' : importance_score}

def Batch_EventClassifier(baseFolderPath):
    # run through all dataset and generate result summary
    result = []
    tankNames = []
    importance_score = np.empty((0,2))
    is0 = np.empty(0,1)
    is1 = np.empty(0, 1)
    is2 = np.empty(0, 1)

    balanced_accuracy = np.empty((0,2))
    for dataPath in tqdm([p for p in baseFolderPath.glob('#*')]):
        data_ = EventClassifier(dataPath)
        tankNames.append(str(dataPath))
        result.append(data_)
        balanced_accuracy = np.vstack([
            balanced_accuracy,
            np.expand_dims(np.array(data_['balanced_accuracy']),0)])
        for i, score in enumerate(data_['importance_score']):
            importance_score = np.vstack([importance_score,
                                          np.expand_dims(np.array([i+1, score]), 0)])

    return {'tankNames' : tankNames, 'result' : result, 'balanced_accuracy' : balanced_accuracy, 'importance_score' : importance_score}
    
output = Batch_EventClassifier(Path(r'E:\EventClassificationDataset'))
print(np.mean(output['balanced_accuracy'],0))
savemat(r'E:\EventClassificationDataset\Output.mat', output)

# 'rgf' kernel : [0.33016645, 0.69981628]
# 'linear' kernel : [0.32275436 0.6936246 ]
#
# plt.plot(np.sum(np.reshape(a.importances_mean, (20, -1)), 0))
#
# rgf로 해도 저 함수 쓰면 어짜피 임폴턴스 구할 수 있음.
# 걍 rgf로 하고
#
# 1. 각 세션마다 cell number 에 따라서 importnace 쭉 값을 나열시키고,
# 2. align 한거 있지 그거에서 좌측에 나타나는 애들, 가운데 나타나는 애들, 우측에 나타나는 애들 범위 3개로 나눠서
# 3. 각각 그룹에 있는 애들의 importance가 어떻게 되는지
# if 확인:
