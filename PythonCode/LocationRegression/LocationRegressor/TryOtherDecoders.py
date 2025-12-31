import re
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from LocationRegressionHelper import *
import time
from numpy.random import default_rng
import argparse
import platform
import numpy as np
from pathlib import Path

# Import all regressors
# Linear regression
from sklearn.linear_model import LinearRegression
# Quadratic regression
from sklearn.preprocessing import PolynomialFeatures
# SVM regressor
from sklearn.svm import SVR
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

def OtherRegressor(tankPath, outputPath, neural_data_rate, truncatedTime_s):
    rng = default_rng()
    numBin = 1
    # Load Tank
    tank_name = re.search('#.*', str(tankPath))[0]
    print(tank_name)

    # Load Data
    neural_data, y_r, y_c, y_deg, midPointTimes, zoneClass = loadData(tankPath, neural_data_rate, truncatedTime_s,
                                                                      removeNestingData=False,
                                                                      removeEncounterData=False,
                                                                      removeWanderData=False,
                                                                      stratifyData=False,
                                                                      doorClosedOnly=False)
    print(neural_data.shape)

    # Dataset Prepared
    X = np.clip(neural_data, -5, 5)
    print('Distance Regressor')
    y = ((y_r - 280) ** 2 + (y_c - 640) ** 2) ** 0.5

    # Prepare array to store regression result from the test dataset
    """
    Total number of column is 3 + (num regressor=4) = 7
    """
    WholeTestResult = np.zeros([X.shape[0], 7])  # num data x [row, col, true, predicted_1, predicted_2, ...
    WholeTestResult[:, :3] = np.hstack((y_r, y_c, y))

    # Prepare array to store test dataset from the unit shuffled test dataset
    numUnit = int(X.shape[1] / numBin)
    print(f'numUnit : {numUnit}')

    # Setup KFold
    CV_split = 5
    kf = StratifiedKFold(n_splits=CV_split, shuffle=True, random_state=622)

    # Start training
    for train_index, test_index in kf.split(X, zoneClass):
        X_train = X[train_index, :]
        X_test = X[test_index, :]

        y_train = y[train_index, :]

        # Linear Regression
        linearReg = LinearRegression()
        linearReg.fit(X_train, y_train)
        WholeTestResult[test_index, 3:4] = linearReg.predict(X_test)
        print("   Linear Finished")

        # Quadratic Regression
        poly = PolynomialFeatures(degree=3)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.fit_transform(X_test)
        quadReg = LinearRegression()
        quadReg.fit(X_train_poly, y_train)
        WholeTestResult[test_index, 4:5] = quadReg.predict(X_test_poly)
        print("   Quadratic Finished")

        # SVM Regressor
        svmReg = SVR()
        svmReg.fit(X_train, np.ravel(y_train))
        WholeTestResult[test_index, 5] = svmReg.predict(X_test)
        print("   SVM Finished")

        # Random Forest Regressor
        rfReg = RandomForestRegressor()
        rfReg.fit(X_train, np.ravel(y_train))
        WholeTestResult[test_index, 6] = rfReg.predict(X_test)
        print("   Random Forest Finished")

    savemat(outputPath / f'{tank_name}result_distance.mat', {
        'WholeTestResult': WholeTestResult,
        'midPointTimes': midPointTimes})


if platform.system() == 'Windows':
    BasePath = Path('D:\Data\Lobster')
else:
    BasePath = Path.home() / 'Data'

InputFolder = BasePath / 'FineDistanceDataset'
OutputFolder = BasePath / 'FineDistanceResult_other_decoders'

for i, tank in enumerate(sorted([p for p in InputFolder.glob('#*')])):
    OtherRegressor(
        tankPath=tank,
        outputPath=OutputFolder,
        neural_data_rate=20,
        truncatedTime_s=10,
    )


