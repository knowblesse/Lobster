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
import seaborn as sns
import matplotlib.pyplot as plt

# Check package version
if (sklearn.__version__ < '0.23.2'):
    raise Exception("scikit-learn package version must be at least 0.23.2")

# Constants
F1_Path = Path('D:\Data\Lobster\Lobster_Recording-200319-161008\TimeAxisAnalysisDataset')
F2_Path = [i for i in F1_Path.iterdir()]
Cond = ['PL', 'IL']

for BASE_PATH in F2_Path:
    for region in Cond:
        REGEXP_CONDITION = '*_'+region+'_*.mat'

        OUTPUT_FILE_PATH = BASE_PATH
        Y_label = ['Head Entry', 'Avoidance Head Withdrawal', 'Escape Head Withdrawal']

        # Variable Setup
        datalist = [i for i in sorted(BASE_PATH.glob(REGEXP_CONDITION))]
        num_data = len(datalist)
        data_name = [i.stem.replace('000,', ',').replace('000]', ']').replace(',', '_') for i in datalist]
        mat_accuracy = np.zeros((num_data, 2))  # left column : shuffled, right column : real
        mat_balanced_accuracy = np.zeros((num_data, 2))
        mat_f1_micro = np.zeros((num_data, 2))
        mat_f1_macro = np.zeros((num_data, 2))
        mat_f1_weighted = np.zeros((num_data, 2))
        mat_confusion = np.zeros((3, 6, num_data))  # [3x3 Shuffled, 3x3 Real] x numdata

        # Function for LOO testing
        def runTest(X, Y):
            Y_pred = np.zeros((len(Y),), dtype='uint8')
            loo = LeaveOneOut()
            print("LOO : Total {:d} tests to perform".format(loo.get_n_splits(X)))
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                clf = SVC(C=2)
                clf.fit(X_train, Y_train)
                Y_pred[test_index] = clf.predict(X_test)
            return Y_pred


        # Run through all datasets
        for datanum, data_path in enumerate(datalist):
            # Load dataset
            data = loadmat(str(data_path.absolute()))
            print(data_path.name + ' is loaded \n')
            X = data.get('X')
            Y = data.get('y')
            Y = np.squeeze(Y)
            Y_real = Y.copy()
            Y_shuffled = Y.copy()
            np.random.shuffle(Y_shuffled)

            Y_pred_shuffled = runTest(X, Y_shuffled)
            Y_pred_real = runTest(X, Y_real)

            score_shuffled = accuracy_score(Y_shuffled, Y_pred_shuffled)
            score_real = accuracy_score(Y_real, Y_pred_real)

            score_shuffled_balanced = balanced_accuracy_score(Y_shuffled, Y_pred_shuffled)
            score_real_balanced = balanced_accuracy_score(Y_real, Y_pred_real)

            score_shuffled_f1_micro = f1_score(Y_shuffled, Y_pred_shuffled, average='micro')
            score_real_f1_micro = f1_score(Y_real, Y_pred_real, average='micro')

            score_shuffled_f1_macro = f1_score(Y_shuffled, Y_pred_shuffled, average='macro')
            score_real_f1_macro = f1_score(Y_real, Y_pred_real, average='macro')

            score_shuffled_f1_weighted = f1_score(Y_shuffled, Y_pred_shuffled, average='weighted')
            score_real_f1_weighted = f1_score(Y_real, Y_pred_real, average='weighted')

            mat_accuracy[datanum, 0] = score_shuffled
            mat_accuracy[datanum, 1] = score_real

            mat_balanced_accuracy[datanum, 0] = score_shuffled_balanced
            mat_balanced_accuracy[datanum, 1] = score_real_balanced

            mat_f1_micro[datanum, 0] = score_shuffled_f1_micro
            mat_f1_micro[datanum, 1] = score_real_f1_micro

            mat_f1_macro[datanum, 0] = score_shuffled_f1_macro
            mat_f1_macro[datanum, 1] = score_real_f1_macro

            mat_f1_weighted[datanum, 0] = score_shuffled_f1_weighted
            mat_f1_weighted[datanum, 1] = score_real_f1_weighted

            mat_confusion[:, 0:3, datanum] = confusion_matrix(Y_shuffled, Y_pred_shuffled)
            mat_confusion[:, 3:, datanum] = confusion_matrix(Y_real, Y_pred_real)
            print('----------{:d}/{:d} finished ----------'.format(datanum + 1, num_data))

        # Save the result into Matlab format
        savemat(OUTPUT_FILE_PATH / ('output_'+region+'.mat'),
                {'dataname': data_name, 'mat_accuracy': mat_accuracy, 'mat_balanced_accuracy': mat_balanced_accuracy,
                'mat_f1_micro': mat_f1_micro, 'mat_f1_macro': mat_f1_macro, 'mat_f1_weighted': mat_f1_weighted, 'mat_confusion': mat_confusion})
