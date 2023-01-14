from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from  LocationRegression.LocationRegressor.LocationRegressionHelper import loadData
from pathlib import Path
from warnings import warn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warn('Set to 225')

def parseAllData(tankName):
    locationDataPath = Path(r"D:/Data/Lobster/FineDistanceDataset") / Path(tankName)
    locationResultPath = Path(r"D:/Data/Lobster/FineDistanceResult") / (tankName + 'result_distance.mat')
    behaviorDataPath = Path(r"D:/Data/Lobster/BehaviorData") / Path(tankName).with_suffix('.mat')
    neural_data, y_r, y_c, midPointTimes = loadData(locationDataPath, neural_data_rate=20, truncatedTime_s=10, removeNestingData=False)
    neural_data = np.clip(neural_data, -5, 5)
    behavior_data = loadmat(behaviorDataPath) #'ParsedData'
    locationResult = loadmat(locationResultPath)
    locationResult = locationResult['WholeTestResult']

    # Parse behavior Data
    ParsedData = behavior_data['ParsedData']
    numTrial = len(ParsedData)
    Trials = np.concatenate(ParsedData[:,0])
    IRs = np.empty((0,2))

    for trialIdx in range(numTrial):
        IRs = np.vstack((IRs, ParsedData[trialIdx,1] + ParsedData[trialIdx,0][0,0]))

    AEResult = [] # 0 for Avoid, 1 for Escape
    for trial, ir, lick, attack in ParsedData:
        nearAttackIRindex = np.where(ir[:, 0] < attack[0, 0])[0][-1] # the last iron which is earlier than attack
        IAttackIROFI = ir[nearAttackIRindex, 1] - attack[0, 0]
        AEResult.append(int(IAttackIROFI >= 0)) # 0 avoid 1 escape

    # Generate Info vectors
    isEncounterZone = np.zeros(midPointTimes.shape, dtype=bool)
    isNestingZone = np.zeros(midPointTimes.shape, dtype=bool)
    inWhichTrial = np.zeros(midPointTimes.shape)
    inAE = np.zeros(midPointTimes.shape)
    for i in range(len(midPointTimes)):
        isEncounterZone[i] = np.any((IRs[:, 0] < midPointTimes[i]) & (midPointTimes[i] < IRs[:, 1]))
        isNestingZone[i] = y_c[i] < 225

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
    # 0 : nesting 1 : foraging 2 : encounter

    return {'neural_data': neural_data,
            'midPointTimes': midPointTimes,
            'ParsedData': ParsedData,
            'Trials': Trials,
            'IRs': IRs,
            'numTrial': numTrial,
            'AEResult': AEResult,
            'zoneClass': zoneClass,
            'inWhichTrial': inWhichTrial,
            'inAE': inAE}
    
def getZoneLDA(neural_data, zoneClass):
    lda = LinearDiscriminantAnalysis()
    lda.fit(neural_data, zoneClass)
    neural_data_transformed = lda.transform(neural_data)

    # Calculate Centroids
    c_nesting = (np.mean(neural_data_transformed[zoneClass==0, 0]), np.mean(neural_data_transformed[zoneClass==0, 1]))
    c_foraging = (np.mean(neural_data_transformed[zoneClass==1, 0]), np.mean(neural_data_transformed[zoneClass==1, 1]))
    c_encounter = (np.mean(neural_data_transformed[zoneClass==2, 0]), np.mean(neural_data_transformed[zoneClass==2, 1]))

    centroids = {'nest': c_nesting, 'foraging': c_foraging, 'encounter': c_encounter}

    return (neural_data_transformed, centroids)

def mahal(points, dataset):
    # return mahalanobis distance
    # row => new points to query
    # col => dimension
    inv_cov = np.linalg.inv(np.cov(dataset, rowvar=False))
    dataset_mean = np.mean(dataset, axis=0)

    output = np.zeros(points.shape[0])

    for i, point in enumerate(points):
        output[i] = np.dot(np.dot(point- dataset_mean, inv_cov), (point-dataset_mean).T) ** 0.5

    return output