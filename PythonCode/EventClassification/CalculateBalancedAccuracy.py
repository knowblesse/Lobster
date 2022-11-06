"""
CalculateBalancedAccuracy
using HEC_Result.mat, get balanced accuracy.
HEC Result has the true and prediction values for HE/HW classifier and A/E Classifier.
"""
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path

# HEC_Result.mat Path
dataPath = Path(r'C:/VCF/Lobster/MatlabCode/ML/HEC_Result.mat')

data = loadmat(dataPath)

PL_shuffled = []
PL_original = []
IL_shuffled = []
IL_original = []

for session in range(40):
    HEHW = data['result'][0][session][0][0][0]
    HE_AE = data['result'][0][session][0][0][1]
    HW_AE = data['result'][0][session][0][0][2]

    numTrial = HE_AE.shape[0]

    # HE Data
    trueData_HE = HE_AE[:,0]
    decodedData_shuffled_HE = 2 * (HEHW[:numTrial, 1] != HEHW[:numTrial, 0]) + HE_AE[:, 1]
    decodedData_original_HE = 2 * (HEHW[:numTrial, 2] != HEHW[:numTrial, 0]) + HE_AE[:, 2]

    #HW Data
    trueData_HW = HW_AE[:,0]
    decodedData_shuffled_HW = -2 * (HEHW[numTrial:, 1] != HEHW[numTrial:, 0]) + HW_AE[:, 1]
    decodedData_original_HW = -2 * (HEHW[numTrial:, 2] != HEHW[numTrial:, 0]) + HW_AE[:, 2]

    # All Data
    trueData = np.concatenate((trueData_HE, trueData_HW))
    decodedData_shuffled = np.concatenate((decodedData_shuffled_HE, decodedData_shuffled_HW))
    decodedData_original = np.concatenate((decodedData_original_HE, decodedData_original_HW))

    acc_shuffled = balanced_accuracy_score(trueData, decodedData_shuffled)
    acc_original = balanced_accuracy_score(trueData, decodedData_original)

    if 'PL' in data['sessionNames'][session]:
        PL_shuffled.append(acc_shuffled)
        PL_original.append(acc_original)
    elif 'IL' in data['sessionNames'][session]:
        IL_shuffled.append(acc_shuffled)
        IL_original.append(acc_original)
