import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Switching.SwitchingHelper import parseAllData, getZoneLDA
from Switching.LDAScripts import *

FolderPath = Path(r'D:/Data/Lobster/FineDistanceDataset')
outputData = []
pbar = tqdm([p for p in FolderPath.glob('#*')])
for tank in pbar:
    tankName = re.search('#.*', str(tank))[0]

    data = parseAllData(tankName)
    zoneClass = data['zoneClass']
    numTrial = data['numTrial']
    midPointTimes = data['midPointTimes']
    Trials = data['Trials']
    AEResult = data['AEResult']
    ParsedData = data['ParsedData']

    neural_data_transformed, centroids = getZoneLDA(data['neural_data'], zoneClass)

    isWanderingInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)
    isReadyInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)

    for trial in np.arange(2, numTrial + 1):
        latency2HeadEntry = ParsedData[trial - 1, 1][0, 0]

        betweenTRON_firstIRON = np.logical_and(
            ParsedData[trial - 1, 0][0, 0] <= midPointTimes,
            midPointTimes < (ParsedData[trial - 1, 1][0, 0] + ParsedData[trial - 1, 0][0, 0])
        )

        # get behavior types
        if latency2HeadEntry >= 5:
            isWanderingInNest = np.logical_or(isWanderingInNest, np.logical_and(betweenTRON_firstIRON, zoneClass == 0))
        else:
            isReadyInNest = np.logical_or(isReadyInNest, np.logical_and(betweenTRON_firstIRON, zoneClass == 0))

    centroids['wanderInNest'] = np.mean(neural_data_transformed[isWanderingInNest, :], 0)
    centroids['readyInNest'] = np.mean(neural_data_transformed[isReadyInNest, :], 0)

    wander_nest = np.sum((np.array(centroids['nest']) - centroids['wanderInNest']) ** 2) ** 0.5
    wander_foraging = np.sum((np.array(centroids['foraging']) - centroids['wanderInNest']) ** 2) ** 0.5

    ready_nest = np.sum((np.array(centroids['nest']) - centroids['readyInNest']) ** 2) ** 0.5
    ready_foraging = np.sum((np.array(centroids['foraging']) - centroids['readyInNest']) ** 2) ** 0.5

    outputData.append([tankName,
                       wander_nest/(wander_nest+wander_foraging),
                       wander_foraging / (wander_nest + wander_foraging),
                       ready_nest / (ready_nest + ready_foraging),
                       ready_nest / (ready_nest + ready_foraging)
                       ])

    # Fact : Neural state of encounter is not similar compared to other states
    # D_nest_foraging = np.sum((np.array(centroids['nest']) - np.array(centroids['foraging'])) ** 2) ** 0.5
    # D_nest_encounter = np.sum((np.array(centroids['nest']) - np.array(centroids['encounter'])) ** 2) ** 0.5
    # D_encounter_foraging = np.sum((np.array(centroids['encounter']) - np.array(centroids['foraging'])) ** 2) ** 0.5
    # D_all = D_nest_foraging + D_nest_encounter + D_encounter_foraging
    # outputData.append([tankName, D_nest_foraging/D_all, D_nest_encounter/D_all, D_encounter_foraging/D_all])

    # Hypothesis : if, neural vector during the nesting area is closer to the "state of encounter zone",
    # then the higher chance of avoidance failure on the following trial

    # # in this part, 'trial' is actual trial number = There is no 0 Trial
    # AvoidData = []
    # EscapeData = []
    # for trial in np.arange(2, numTrial + 1):
    #     betweenTrials = np.logical_and(Trials[trial - 2, 1] <= midPointTimes, midPointTimes < Trials[trial - 1, 0])
    #     targetIndex = np.logical_and(zoneClass == 0, betweenTrials)
    #     if np.sum(targetIndex) == 0:
    #         continue
    #     distance2encounterState = np.mean(
    #         np.sum((centroids['encounter'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     distance2nestState = np.mean(
    #         np.sum((centroids['nest'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     vec = (distance2encounterState - distance2nestState) / distance2nestState
    #     if AEResult[trial - 1] == 0:  # current Trial's result is avoid
    #         AvoidData.append(vec)
    #     else:
    #         EscapeData.append(vec)
    # if (len(AvoidData) > 5) and (len(EscapeData) > 5):
    #     outputData.append([tankName, np.mean(AvoidData), np.mean(EscapeData)])
    #
    # # Hypothesis : if, neural vector during the nesting area is closer to the "state of encounter zone",
    # # then the higher chance of avoidance failure on the following trial
    #
    # # # in this part, 'trial' is actual trial number = There is no 0 Trial
    # AvoidData = []
    # EscapeData = []
    # for trial in np.arange(2, numTrial + 1):
    #     betweenTrials = np.logical_and(Trials[trial - 2, 1] <= midPointTimes, midPointTimes < Trials[trial - 1, 0])
    #     targetIndex = np.logical_and(zoneClass == 0, betweenTrials)
    #     if np.sum(targetIndex) == 0:
    #         continue
    #     distance2encounterState = np.mean(
    #         np.sum((centroids['encounter'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     distance2nestState = np.mean(
    #         np.sum((centroids['nest'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     vec = (distance2encounterState - distance2nestState) / distance2nestState
    #     if AEResult[trial - 1] == 0:  # current Trial's result is avoid
    #         AvoidData.append(vec)
    #     else:
    #         EscapeData.append(vec)
    # if (len(AvoidData) > 5) and (len(EscapeData) > 5):
    #     outputData.append([tankName, np.mean(AvoidData), np.mean(EscapeData)])
np.savetxt('C:/Users/Knowblesse/Desktop/stds.csv', np.array(outputData), delimiter=',', fmt='%s')
