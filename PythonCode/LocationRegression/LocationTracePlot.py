from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

TANK_location = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1\#20JUN1-200923-142219_PL')
X_location = next(TANK_location.glob('*regressionData_X.csv'))
y_location = next(TANK_location.glob('*regressionData_y.csv'))

X = np.loadtxt(str(X_location), delimiter=',')
y = np.loadtxt(str(y_location), delimiter=',')

y_decoded = np.zeros((0,3))

kf = KFold(n_splits=5)  # 0.2 testing set

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg1 = MLPRegressor(hidden_layer_sizes=(200, 50), max_iter=1000, learning_rate_init=0.01)
    reg2 = MLPRegressor(hidden_layer_sizes=(200, 50), max_iter=1000, learning_rate_init=0.01)
    reg1.fit(X_train, y_train[:, 0])
    reg2.fit(X_train, y_train[:, 1])

    y_decoded = np.vstack((y_decoded,
                           np.hstack((np.expand_dims(test_index, axis=1),
                                      np.vstack((reg1.predict(X_test), reg2.predict(X_test))).T
                                      ))
                           ))

