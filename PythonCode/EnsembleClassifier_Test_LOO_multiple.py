# EnsembleClassifier_Test_LOO
# Ensemble Classifier using Leave One Out method
import os
import  numpy as np
import sklearn
if (sklearn.__version__ != '0.23.2'):
    raise Exception("scikit-learn package version must be 0.23.2")
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

from scipy.io import loadmat, savemat
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
BASE_PATH = r'C:\VCF\Lobster\data\GR7\[-8000,-6000]'
#BASE_PATH = r'C:\VCF\Lobster\data\20JUN1\[+1000,+3000]'
DATASET_NAME = 'GR7_[-8000,-6000].mat'
#DATASET_NAME = '20JUN1_[+1000,+3000].mat'
Y_label = ['Head Entry', 'Avoidance', 'Escape']

# Variable Setup
datalist = os.listdir(BASE_PATH)
num_data = len(datalist)
mat_accuracy = np.zeros((num_data,2)) # left column : shuffled, right column : real
mat_confusion = np.zeros((3,6,num_data)) # [3x3 Shuffled, 3x3 Real] x numdata

# Function for LOO testing
def runTest(X,Y):
    Y_pred = np.zeros((len(Y),), dtype='uint8')
    loo = LeaveOneOut()
    print("LOO : Total {:d} tests to perform".format(loo.get_n_splits(X)))
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf = SVC(C=2, gamma='scale')
        clf.fit(X_train, Y_train)
        Y_pred[test_index] = clf.predict(X_test)
    return Y_pred

# Run through all datasets
for datanum in np.arange(num_data):
    # Load dataset
    data = loadmat(os.path.join(BASE_PATH, datalist[datanum]))
    print(datalist[datanum] + ' is loaded \n')
    X = data.get('X')
    Y = data.get('y')
    Y = np.squeeze(Y)
    Y_real = Y.copy()
    Y_shuffled = Y.copy()
    np.random.shuffle(Y_shuffled)

    Y_pred_shuffled = runTest(X, Y_shuffled)
    Y_pred_real = runTest(X,Y_real)

    score_shuffled = accuracy_score(Y_shuffled, Y_pred_shuffled)
    score_real = accuracy_score(Y_real, Y_pred_real)

    mat_accuracy[datanum,0] = score_shuffled
    mat_accuracy[datanum,1] = score_real

    mat_confusion[:,0:3,datanum] = confusion_matrix(Y_shuffled, Y_pred_shuffled)
    mat_confusion[:,3:,datanum] = confusion_matrix(Y_real, Y_pred_real)
    print('----------{:d}/{:d} finished ----------'.format(datanum+1, num_data))

# # Draw End result
# cmap = sns.color_palette("light:g", as_cmap=True)
# f = plt.figure(2, figsize=(12, 6))
# f.clear()
# ax1, ax2 = f.subplots(1,2)
# f.suptitle('IL',fontsize = 25)
# confusion_mat_shuffled = mat_confusion[:,0:3,:]
# confusion_mat_real = mat_confusion[:,3:,:]
# num_sample = np.sum(np.sum(confusion_mat_shuffled,axis=2),axis=1).reshape(-1,1)
#
# sns.heatmap(np.sum(confusion_mat_shuffled,axis=2) / num_sample, ax=ax1, cmap=cmap, vmin=0, vmax=1, annot=True, annot_kws={'fontsize':15,'color':'k'}, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# sns.heatmap(np.sum(confusion_mat_real,axis=2) / num_sample, ax=ax2, cmap=cmap, vmin=0, vmax=1, annot=True, annot_kws={'fontsize':15,'color':'k'}, square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
# # present std
# confusion_mat_std_shuffled = np.std(confusion_mat_shuffled,axis=2) / num_sample
# confusion_mat_std_real = np.std(confusion_mat_real,axis=2) / num_sample
# for i in np.arange(3):
#     for j in np.arange(3):
#         ax1.text(0.5 + i, 0.6 + j,'(±{:.3f})'.format(confusion_mat_std_shuffled[i,j]),verticalalignment='top',horizontalalignment='center',color='w')
#         ax2.text(0.5 + i, 0.6 + j,'(±{:.3f})'.format(confusion_mat_std_real[i,j]), verticalalignment = 'top', horizontalalignment = 'center', color = 'w')
#
# ax1.set_title('shuffled : {:5.3f}%(±{:5.3f})'.format(np.mean(mat_accuracy,axis=0)[0]*100, np.std(mat_accuracy[:,0]*100)),fontsize=18)
# ax1.set_xticklabels(Y_label,verticalalignment='center')
# ax1.set_yticklabels(Y_label,verticalalignment='center')
# ax1.set_xlabel('predicted',fontsize=15)
# ax1.set_ylabel('actual',labelpad=10,fontsize=15)
# ax2.set_title('real : {:5.3f}%(±{:5.3f})'.format(np.mean(mat_accuracy,axis=0)[1]*100, np.std(mat_accuracy[:,1]*100)),fontsize=19)
# ax2.set_xticklabels(Y_label,verticalalignment='center')
# ax2.set_yticklabels(Y_label,verticalalignment='center')
# ax2.set_xlabel('predicted',fontsize=15)
# ax2.set_ylabel('actual',labelpad=10,fontsize=15)
# plt.show()

# Save the result into Matlab format
savemat(DATASET_NAME,{'mat_accuracy' : mat_accuracy, 'mat_confusion' : mat_confusion})
