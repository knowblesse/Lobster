# EnsembleClassifier_Test_LOO
# Ensemble Classifier using Leave One Out method graph sample
import os
import  numpy as np
import sklearn
if (sklearn.__version__ != '0.23.2'):
    raise Exception("scikit-learn package version must be 0.23.2")
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Data to load and analyze
datanum = 6

# Load .mat data
BASE_PATH = r'C:\Users\Knowblesse\SynologyDrive\20JUN'
datalist = os.listdir(BASE_PATH)
data = loadmat(os.path.join(BASE_PATH, datalist[datanum]))
print(datalist[datanum] + ' is loaded \n')
X = data.get('X')
Y = data.get('y')
Y = np.squeeze(Y)
Y_shuffled = Y.copy()
np.random.shuffle(Y_shuffled)
Y_label = ['Head Entry', 'Avoidance', 'Escape']

# Model Generation
from sklearn.svm import SVC

# Leave One Out Loop
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

print('Testing for real data')

def runTest(X,Y):
    Y_pred = np.zeros((len(Y),), dtype='uint8')
    loo = LeaveOneOut()
    print("Total %d Test is performed"%loo.get_n_splits(X))
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        clf = SVC(C=2, gamma='scale')
        clf.fit(X_train, Y_train)
        Y_pred[test_index] = clf.predict(X_test)
    return Y_pred
Y_pred = runTest(X,Y)
Y_shuffled_pred = runTest(X, Y_shuffled)

print('Accuracy Score')
score_shuffled = accuracy_score(Y_shuffled, Y_shuffled_pred)
score_real = accuracy_score(Y, Y_pred)
print(f'Shuffled : {score_shuffled}')
print(f'Real     : {score_real}')

confusion_mat = confusion_matrix(Y, Y_pred,normalize='true') # row is actual. # column is predicted
confusion_mat_shuffled = confusion_matrix(Y_shuffled, Y_shuffled_pred,normalize='true') # row is actual. # column is predicted
cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
f = plt.figure(1, figsize=(12, 6))
f.clear()
ax1, ax2 = f.subplots(1,2)
sns.heatmap(confusion_mat_shuffled, ax=ax1, cmap=cmap, vmin=0, vmax=1, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=Y_label, yticklabels=Y_label)
sns.heatmap(confusion_mat, ax=ax2, cmap=cmap, vmin=0, vmax=1, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=Y_label, yticklabels=Y_label)
ax1.set_title('shuffled : {:5.3f}'.format(score_shuffled))
ax1.set_xlabel('predicted')
ax1.set_ylabel('actual')
ax2.set_title('real : {:5.3f}'.format(score_real))
ax2.set_xlabel('predicted')
ax2.set_ylabel('actual')
plt.show()

