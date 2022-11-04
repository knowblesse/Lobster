from scipy.io import loadmat
import re
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from  LocationRegression.LocationRegressor.LocationRegressionHelper import loadData
from pathlib import Path
import os

tankName = '#21JAN2-210428-195618_IL'
locationDataPath = Path(r"D:/Data/Lobster/LocationRegressionData") / Path(tankName)
behaviorDataPath = Path(r"D:/Data/Lobster/BehaviorData") / Path(tankName).with_suffix('.mat')
neural_data, y_r, y_c, midPointTimes = loadData(locationDataPath, neural_data_rate=2, truncatedTime_s=10, removeNestingData=False)
neural_data = np.clip(neural_data, -5, 5)
behavior_data = loadmat(behaviorDataPath)
#'Attacks', 'IRs', 'Licks', 'ParsedData', 'Trials'

# Parse behavior Data and Generate Class Vectors
IRs = behavior_data['IRs']

isEncounterZone = np.zeros(midPointTimes.shape, dtype=bool)
isNestingZone = np.zeros(midPointTimes.shape, dtype=bool)

for i in range(len(midPointTimes)):
    isEncounterZone[i] = np.any((IRs[:, 0] < midPointTimes[i]) & (midPointTimes[i] < IRs[:, 1]))
    isNestingZone[i] = y_c[i] < 200

zoneClass = (~isNestingZone).astype(int) + isEncounterZone.astype(int)
# 0 : nesting
# 1 : foraging
# 2 : encounter

# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

neural_data_transformed = pca.fit_transform(neural_data)
fig1 = plt.figure(1)
fig1.clf()
ax1 = fig1.subplots(1,1)
ax1.scatter(neural_data_transformed[zoneClass==0, 0], neural_data_transformed[zoneClass==0, 1], c='r', s=4)
ax1.scatter(neural_data_transformed[zoneClass==1, 0], neural_data_transformed[zoneClass==1, 1], c='b', s=4)
ax1.scatter(neural_data_transformed[zoneClass==2, 0], neural_data_transformed[zoneClass==2, 1], c='g', s=4)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(neural_data, zoneClass)
neural_data_transformed = lda.transform(neural_data)

fig2 = plt.figure(2)
fig2.clf()
ax2 = fig2.subplots(1,1)
ax2.scatter(neural_data_transformed[zoneClass==0, 0], neural_data_transformed[zoneClass==0, 1], c='r', s=4)
ax2.scatter(neural_data_transformed[zoneClass==1, 0], neural_data_transformed[zoneClass==1, 1], c='b', s=4)
ax2.scatter(neural_data_transformed[zoneClass==2, 0], neural_data_transformed[zoneClass==2, 1], c='g', s=4)
ax2.legend(["Nesting", "Foraging", "Encounter"])

