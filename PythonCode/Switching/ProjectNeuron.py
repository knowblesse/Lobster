from scipy.io import loadmat
import re
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from  LocationRegression.LocationRegressor.LocationRegressionHelper import loadData
from pathlib import Path
import os
from warnings import warn

tankName = '#21AUG3-211028-165958_PL'
locationDataPath = Path(r"D:/Data/Lobster/LocationRegressionData") / Path(tankName)
behaviorDataPath = Path(r"D:/Data/Lobster/BehaviorData") / Path(tankName).with_suffix('.mat')
neural_data, y_r, y_c, midPointTimes = loadData(locationDataPath, neural_data_rate=2, truncatedTime_s=10, removeNestingData=False)
neural_data = np.clip(neural_data, -5, 5)
behavior_data = loadmat(behaviorDataPath)
#'Attacks', 'IRs', 'Licks', 'ParsedData', 'Trials'

# Parse behavior Data
ParsedData = behavior_data['ParsedData']
numTrial = len(ParsedData)
Trials = np.concatenate(ParsedData[:,0])
IRs = np.concatenate(ParsedData[:,1])

AEResult = []
for trial, ir, lick, attack in ParsedData:
    nearAttackIRindex = np.where(ir[:, 0] < attack[0, 0])[0][-1] # the last iron which is earlier than attack
    IAttackIROFI = ir[nearAttackIRindex, 1] - attack[0, 0]
    AEResult.append(int(IAttackIROFI >= 0)) # 0 avoid 1 escape

# Generate Class Vectors

isEncounterZone = np.zeros(midPointTimes.shape, dtype=bool)
isNestingZone = np.zeros(midPointTimes.shape, dtype=bool)
inWhichTrial = np.zeros(midPointTimes.shape)
inAE = np.zeros(midPointTimes.shape)
for i in range(len(midPointTimes)):
    isEncounterZone[i] = np.any((IRs[:, 0] < midPointTimes[i]) & (midPointTimes[i] < IRs[:, 1]))
    isNestingZone[i] = y_c[i] < 200
    idx = np.where(Trials[:,0] > midPointTimes[i])[0] # cf. trial start from 1. 0 trial means activation before the first trial begins
    if len(idx) == 0:
        idx = Trials.shape[0]
    else:
        idx = idx[0]
    inWhichTrial[i] = idx
    if idx == 0:
        inAE[i] = -1
        warn('0 trials exist')
    inAE[i] = AEResult[idx-1]

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
ax1.set_title(tankName + "- PCA")

# LDA

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
ax2.set_title(tankName + "- LDA")

fig21 = plt.figure(21)
fig21.clf()
ax21 = fig21.subplots(1,1)
ax21.scatter(neural_data_transformed[np.logical_and(zoneClass==0, inAE==0), 0], neural_data_transformed[np.logical_and(zoneClass==0, inAE==0), 1], c='r', s=18, marker='o', alpha=0.5)
ax21.scatter(neural_data_transformed[np.logical_and(zoneClass==0, inAE==1), 0], neural_data_transformed[np.logical_and(zoneClass==0, inAE==1), 1], c='g', s=18, marker='o', alpha=0.5)
ax21.scatter(neural_data_transformed[np.logical_and(zoneClass==1, inAE==0), 0], neural_data_transformed[np.logical_and(zoneClass==1, inAE==0), 1], c='r', s=18, marker='^', alpha=0.5)
ax21.scatter(neural_data_transformed[np.logical_and(zoneClass==1, inAE==1), 0], neural_data_transformed[np.logical_and(zoneClass==1, inAE==1), 1], c='g', s=18, marker='^', alpha=0.5)
ax21.scatter(neural_data_transformed[np.logical_and(zoneClass==2, inAE==0), 0], neural_data_transformed[np.logical_and(zoneClass==2, inAE==0), 1], c='r', s=18, marker='s', alpha=0.5)
ax21.scatter(neural_data_transformed[np.logical_and(zoneClass==2, inAE==1), 0], neural_data_transformed[np.logical_and(zoneClass==2, inAE==1), 1], c='g', s=18, marker='s', alpha=0.5)

# Calculate Centroid for Encounter neural vector
lda_encounter = LinearDiscriminantAnalysis()
neural_data_transformed_lda = lda_encounter.fit_transform(neural_data, isEncounterZone)

fig3 = plt.figure(3)
fig3.clf()
ax3 = fig3.subplots(1,1)
ax3.scatter(neural_data_transformed_lda[isEncounterZone==0], np.ones(np.sum(isEncounterZone==0)), c='r', s=6, )
ax3.scatter(neural_data_transformed_lda[isEncounterZone==1, 0], 2*np.ones(np.sum(isEncounterZone==1)), c='b', s=6)
ax3.scatter(np.mean(neural_data_transformed_lda[isEncounterZone==0]), 1, c='k', marker='*')
ax3.scatter(np.mean(neural_data_transformed_lda[isEncounterZone==1]), 2, c='k', marker='*')
ax3.legend(["Outside Encounter zone", "Inside Encounter zone"])

